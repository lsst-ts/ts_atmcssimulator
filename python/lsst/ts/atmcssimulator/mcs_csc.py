# This file is part of ts_atmcssimulator.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["ATMCSCsc", "Axis", "MainAxes", "run_atmcs_simulator"]

import asyncio
import enum

import numpy as np
from lsst.ts import salobj, simactuators, utils
from lsst.ts.idl.enums.ATMCS import AtMountState, M3ExitPort, M3State

from . import __version__


class Axis(enum.IntEnum):
    Elevation = 0
    Azimuth = 1
    NA1 = 2
    NA2 = 3
    M3 = 4


MainAxes = (Axis.Elevation, Axis.Azimuth, Axis.NA1, Axis.NA2)


class ATMCSCsc(salobj.BaseCsc):
    """Simulator for auxiliary telescope motor control system CSC.

    Parameters
    ----------
    initial_state : `salobj.State` or `int` (optional)
        The initial state of the CSC. This is provided for unit testing,
        as real CSCs should start up in `State.STANDBY`, the default.

    Notes
    -----
    .. _axis:

    The axes are, in order:
    - elevation
    - azimuth
    - Nasmyth1 rotator
    - Nasmyth2 rotator
    - m3 rotator

    **Limitations**

    * Jerk is infinite.
    * When an axis has multiple motors or encoders, all are treated as
      identical (e.g. report identical positions).
    * The only way to hit a limit switch is to configure the position
      of that switch within the command limits for that axis.
    * The CSC always wakes up at position 0 for all axes except elevation,
      and at the minimum allowed position for elevation.
    * The model for the azimuth topple block is primitive:

      * The CCW switch is active if az < az1
      * The CW switch is  active if az > az2
      * Otherwise neither switch is active
    """

    valid_simulation_modes = [1]
    # Append "-sim" to avoid confusion with the real ATMCS CSC.
    version = f"{__version__}-sim"

    def __init__(self, initial_state=salobj.State.STANDBY):
        super().__init__(
            name="ATMCS", index=0, initial_state=initial_state, simulation_mode=1
        )
        # interval between telemetry updates (sec)
        self._telemetry_interval = 1
        # number of event updates per telemetry update
        self._events_per_telemetry = 10
        # task that runs while the events_and_telemetry_loop runs
        self._events_and_telemetry_task = utils.make_done_future()
        # task that runs while axes are slewing to a halt from stopTracking
        self._stop_tracking_task = utils.make_done_future()
        # task that runs while axes are halting before being disabled
        self._disable_all_drives_task = utils.make_done_future()
        # Dict of M3ExitPort (the instrument port M3 points to): tuple of:
        # * index of self.m3_port_positions: the M3 position for this port
        # * M3State (state of M3 axis when pointing to this port)
        # * Rotator axis at this port, as an Axis enum,
        #   or None if this port has no rotator.
        self._port_info_dict = {
            M3ExitPort.NASMYTH1: (0, M3State.NASMYTH1, Axis.NA1),
            M3ExitPort.NASMYTH2: (1, M3State.NASMYTH2, Axis.NA2),
            M3ExitPort.PORT3: (2, M3State.PORT3, None),
        }
        # Name of minimum limit switch event for each axis.
        self._min_lim_names = (
            "elevationLimitSwitchLower",
            "azimuthLimitSwitchCW",
            "nasmyth1LimitSwitchCW",
            "nasmyth2LimitSwitchCW",
            "m3RotatorLimitSwitchCW",
        )
        # Name of maximum limit switch event for each axis.
        self._max_lim_names = (
            "elevationLimitSwitchUpper",
            "azimuthLimitSwitchCCW",
            "nasmyth1LimitSwitchCCW",
            "nasmyth2LimitSwitchCCW",
            "m3RotatorLimitSwitchCCW",
        )
        # Name of "in position" event for each axis,
        # excluding ``allAxesInPosition``.
        self._in_position_names = (
            "elevationInPosition",
            "azimuthInPosition",
            "nasmyth1RotatorInPosition",
            "nasmyth2RotatorInPosition",
            "m3InPosition",
        )
        # Name of drive status events for each axis.
        self._drive_status_names = (
            ("elevationDriveStatus",),
            (
                "azimuthDrive1Status",
                "azimuthDrive2Status",
            ),
            ("nasmyth1DriveStatus",),
            ("nasmyth2DriveStatus",),
            ("m3DriveStatus",),
        )
        # Name of brake events for each axis.
        self._brake_names = (
            ("elevationBrake",),
            (
                "azimuthBrake1",
                "azimuthBrake2",
            ),
            ("nasmyth1Brake",),
            ("nasmyth2Brake",),
            (),
        )
        # Has tracking been enabled by startTracking?
        # This remains true until stopTracking is called or the
        # summary state is no longer salobj.State.Enabled,
        # even if some drives have been disabled by running into limits.
        self._tracking_enabled = False
        # Is this particular axis enabled?
        # This remains true until stopTracking is called or the
        # summary state is no longer salobj.State.Enabled,
        # or the axis runs into a limit.
        # Note that the brakes automatically come on/off
        # if the axis is disabled/enabled, respectively.
        self._axis_enabled = np.zeros([5], dtype=bool)
        # Timer to kill tracking if trackTarget doesn't arrive in time.
        self._kill_tracking_timer = utils.make_done_future()

    async def close_tasks(self):
        await super().close_tasks()
        self._disable_all_drives_task.cancel()
        self._stop_tracking_task.cancel()
        self._events_and_telemetry_task.cancel()
        self._kill_tracking_timer.cancel()

    async def start(self):
        await super().start()
        await self.configure()

    async def configure(
        self,
        max_tracking_interval=2.5,
        min_commanded_position=(5, -270, -165, -165, 0),
        max_commanded_position=(90, 270, 165, 165, 180),
        start_position=(80, 0, 0, 0, 0),
        min_limit_switch_position=(3, -272, -167, -167, -2),
        max_limit_switch_position=(92, 272, 167, 167, 182),
        max_velocity=(5, 5, 5, 5, 5),
        max_acceleration=(3, 3, 3, 3, 3),
        topple_azimuth=(2, 5),
        m3_port_positions=(0, 180, 90),
        needed_in_pos=3,
        axis_encoder_counts_per_deg=(3.6e6, 3.6e6, 3.6e6, 3.6e6, 3.6e6),
        motor_encoder_counts_per_deg=(3.6e5, 3.6e5, 3.6e5, 3.6e5, 3.6e5),
        motor_axis_ratio=(100, 100, 100, 100, 100),
        torque_per_accel=(1, 1, 1, 1, 1),
        nsettle=2,
        limit_overtravel=1,
    ):
        """Set configuration.

        Parameters
        ----------
        max_tracking_interval : `float`
            Maximum time between tracking updates (sec)
        min_commanded_position : ``iterable`` of 5 `float`
            Minimum commanded position for each axis, in deg
        start_position :  : ``iterable`` of 5 `float`
            Initial position for each axis, in deg
        max_commanded_position : ``iterable`` of 5 `float`
            Minimum commanded position for each axis, in deg
        min_limit_switch_position : ``iterable`` of 5 `float`
            Position of minimum L1 limit switch for each axis, in deg
        max_limit_switch_position : ``iterable`` of 5 `float`
            Position of maximum L1 limit switch for each axis, in deg
        max_velocity : ``iterable`` of 5 `float`
            Maximum velocity of each axis, in deg/sec
        max_acceleration : ``iterable`` of 5 `float`
            Maximum acceleration of each axis, in deg/sec
        topple_azimuth : ``iterable`` of 2 `float`
            Min, max azimuth at which the topple block moves, in deg
        m3_port_positions : ``iterable`` of 3 `float`
            M3 position of instrument ports NA1, NA2 and Port3,
            in that order.
        axis_encoder_counts_per_deg : `list` [`float`]
            Axis encoder resolution, for each axis, in counts/deg
        motor_encoder_counts_per_deg : `list` [`float`]
            Motor encoder resolution, for each axis, in counts/deg
        motor_axis_ratio : `list` [`float`]
            Number of turns of the motor for one turn of the axis.
        torque_per_accel :  `list` [`float`]
            Motor torque per unit of acceleration,
            in units of measuredTorque/(deg/sec^2)
        nsettle : `int`
            Number of consecutive trackPosition commands that result in
            a tracking path before we report an axis is tracking.
        limit_overtravel : `float`
            Distance from limit switches to hard stops (deg).

        Raises
        ------
        lsst.ts.salobj.ExpectedError
            If any list argument has the wrong number of values,
            or if any value cannot be cast to float.
        lsst.ts.salobj.ExpectedError
            If any max_velocity or max_acceleration value <= 0.
        lsst.ts.salobj.ExpectedError
            If min_commanded_position > max_commanded_position
            or start_position > min_commanded_position
            or start_position > max_commanded_position for any axis.
        lsst.ts.salobj.ExpectedError
            If limit_overtravel < 0.
        """

        def convert_values(name, values, nval):
            out = np.array(values, dtype=float)
            if out.shape != (nval,):
                raise salobj.ExpectedError(
                    f"Could not format {name}={values!r} as {nval} floats"
                )
            return out

        # convert and check all values first,
        # so nothing changes if any input is invalid
        min_commanded_position = convert_values(
            "min_commanded_position", min_commanded_position, 5
        )
        max_commanded_position = convert_values(
            "max_commanded_position", max_commanded_position, 5
        )
        start_position = convert_values("start_position", start_position, 5)
        for axis in Axis:
            if max_commanded_position[axis] < min_commanded_position[axis]:
                raise salobj.ExpectedError(
                    f"max_commanded_position[{axis}]={max_commanded_position[axis]} <= "
                    f"min_commanded_position[{axis}]={min_commanded_position[axis]}"
                )
            if min_commanded_position[axis] > start_position[axis]:
                raise salobj.ExpectedError(
                    f"min_commanded_position[{axis}]={min_commanded_position[axis]} > "
                    f"start_position[{axis}]={start_position[axis]}"
                )
            if max_commanded_position[axis] < start_position[axis]:
                raise salobj.ExpectedError(
                    f"max_commanded_position[{axis}]={max_commanded_position[axis]} < "
                    f"start_position[{axis}]={start_position[axis]}"
                )

        min_limit_switch_position = convert_values(
            "min_limit_switch_position", min_limit_switch_position, 5
        )
        max_limit_switch_position = convert_values(
            "max_limit_switch_position", max_limit_switch_position, 5
        )
        max_velocity = convert_values("max_velocity", max_velocity, 5)
        max_acceleration = convert_values("max_acceleration", max_acceleration, 5)
        if max_velocity.min() <= 0:
            raise salobj.ExpectedError(
                f"max_velocity={max_velocity}; all values must be positive"
            )
        if max_acceleration.min() <= 0:
            raise salobj.ExpectedError(
                f"max_acceleration={max_acceleration}; all values must be positive"
            )
        topple_azimuth = convert_values("topple_azimuth", topple_azimuth, 2)
        m3_port_positions = convert_values("m3_port_positions", m3_port_positions, 3)
        axis_encoder_counts_per_deg = convert_values(
            "axis_encoder_counts_per_deg", axis_encoder_counts_per_deg, 5
        )
        motor_encoder_counts_per_deg = convert_values(
            "motor_encoder_counts_per_deg", motor_encoder_counts_per_deg, 5
        )
        motor_axis_ratio = convert_values("motor_axis_ratio", motor_axis_ratio, 5)
        torque_per_accel = convert_values("torque_per_accel", torque_per_accel, 5)
        if limit_overtravel < 0:
            raise salobj.ExpectedError(
                f"limit_overtravel={limit_overtravel} must be >= 0"
            )

        self.max_tracking_interval = max_tracking_interval
        self.min_commanded_position = min_commanded_position
        self.max_commanded_position = max_commanded_position
        self.min_limit_switch_position = min_limit_switch_position
        self.max_limit_switch_position = max_limit_switch_position
        self.max_velocity = max_velocity
        self.topple_azimuth = topple_azimuth
        self.m3_port_positions = m3_port_positions
        self.axis_encoder_counts_per_deg = axis_encoder_counts_per_deg
        self.motor_encoder_counts_per_deg = motor_encoder_counts_per_deg
        self.motor_axis_ratio = motor_axis_ratio
        self.torque_per_accel = torque_per_accel
        self.nsettle = nsettle
        # allowed position error for M3 to be considered in position (deg)
        self.m3tolerance = 1e-5
        self.limit_overtravel = limit_overtravel

        tai = utils.current_tai()
        self.actuators = [
            simactuators.TrackingActuator(
                min_position=self.min_commanded_position[axis],
                max_position=self.max_commanded_position[axis],
                max_velocity=max_velocity[axis],
                max_acceleration=max_acceleration[axis],
                # Use 0 for M3 to prevent tracking.
                dtmax_track=0 if axis == 4 else self.max_tracking_interval,
                nsettle=self.nsettle,
                tai=tai,
                start_position=start_position[axis],
            )
            for axis in Axis
        ]
        self.actuators[0].verbose = True

        await self.evt_positionLimits.set_write(
            minimum=min_commanded_position,
            maximum=max_commanded_position,
            force_output=True,
        )

    async def do_startTracking(self, data):
        self.assert_enabled("startTracking")
        if not self.evt_m3InPosition.data.inPosition:
            raise salobj.ExpectedError(
                "Cannot startTracking until M3 is at a known position"
            )
        if not self._stop_tracking_task.done():
            raise salobj.ExpectedError("stopTracking not finished yet")
        self._tracking_enabled = True
        await self.update_events()
        self._set_tracking_timer(restart=True)

    async def do_trackTarget(self, data):
        self.assert_enabled("trackTarget")
        if not self._tracking_enabled:
            raise salobj.ExpectedError("Cannot trackTarget until tracking is enabled")
        try:
            position = np.array(
                [
                    data.elevation,
                    data.azimuth,
                    data.nasmyth1RotatorAngle,
                    data.nasmyth2RotatorAngle,
                ],
                dtype=float,
            )
            velocity = np.array(
                [
                    data.elevationVelocity,
                    data.azimuthVelocity,
                    data.nasmyth1RotatorAngleVelocity,
                    data.nasmyth2RotatorAngleVelocity,
                ],
                dtype=float,
            )
            dt = utils.current_tai() - data.taiTime
            current_position = position + dt * velocity
            if np.any(current_position < self.min_commanded_position[0:4]) or np.any(
                current_position > self.max_commanded_position[0:4]
            ):
                raise salobj.ExpectedError(
                    f"One or more target positions {current_position} not in range "
                    f"{self.min_commanded_position} to {self.max_commanded_position} "
                    "at the current time"
                )
            if np.any(np.abs(velocity) > self.max_velocity[0:4]):
                raise salobj.ExpectedError(
                    "Magnitude of one or more target velocities "
                    f"{velocity} > {self.max_velocity}"
                )
        except Exception as e:
            await self.fault(code=1, report=f"trackTarget failed: {e}")
            raise

        for i in range(4):
            self.actuators[i].set_target(
                tai=data.taiTime, position=position[i], velocity=velocity[i]
            )

        target_fields = (
            "azimuth",
            "azimuthVelocity",
            "elevation",
            "elevationVelocity",
            "nasmyth1RotatorAngle",
            "nasmyth1RotatorAngleVelocity",
            "nasmyth2RotatorAngle",
            "nasmyth2RotatorAngleVelocity",
            "taiTime",
            "trackId",
            "tracksys",
            "radesys",
        )
        evt_kwargs = dict((field, getattr(data, field)) for field in target_fields)
        await self.evt_target.set_write(**evt_kwargs, force_output=True)
        self.tel_mount_AzEl_Encoders.set(trackId=data.trackId)
        self.tel_mount_Nasmyth_Encoders.set(trackId=data.trackId)

        self._set_tracking_timer(restart=True)

    def _set_tracking_timer(self, restart):
        """Restart or stop the tracking timer.

        Parameters
        ----------
        restart : `bool`
            If True then start or restart the tracking timer, else stop it.
        """
        self._kill_tracking_timer.cancel()
        if restart:
            self._kill_tracking_timer = asyncio.ensure_future(self.kill_tracking())

    async def do_setInstrumentPort(self, data):
        self.assert_enabled("setInstrumentPort")
        if self._tracking_enabled:
            raise salobj.ExpectedError(
                "Cannot setInstrumentPort while tracking is enabled"
            )
        port = data.port
        try:
            m3_port_positions_ind = self._port_info_dict[port][0]
        except IndexError:
            raise salobj.ExpectedError(f"Invalid port={port}")
        try:
            m3_port_positions = self.m3_port_positions[m3_port_positions_ind]
        except IndexError:
            raise RuntimeError(
                f"Bug! invalid m3_port_positions_ind={m3_port_positions_ind} for port={port}"
            )
        await self.evt_m3PortSelected.set_write(selected=port)
        m3actuator = self.actuators[Axis.M3]
        if (
            m3actuator.target.position == m3_port_positions
            and self.evt_m3InPosition.data.inPosition
        ):
            # already there; don't do anything
            return
        self.actuators[Axis.M3].set_target(
            tai=utils.current_tai(), position=m3_port_positions, velocity=0
        )
        self._axis_enabled[Axis.NA1] = False
        self._axis_enabled[Axis.NA2] = False
        await self.update_events()

    async def do_stopTracking(self, data):
        self.assert_enabled("stopTracking")
        if not self._stop_tracking_task.done():
            raise salobj.ExpectedError("Already stopping")
        self._set_tracking_timer(restart=False)
        self._tracking_enabled = False
        for axis in MainAxes:
            self.actuators[axis].stop()
        self._stop_tracking_task.cancel()
        self._stop_tracking_task = asyncio.ensure_future(self._finish_stop_tracking())
        await self.update_events()

    async def kill_tracking(self):
        """Wait ``self.max_tracking_interval`` seconds and disable tracking.

        Intended for use by `do_trackTarget` to abort tracking
        if the next ``trackTarget`` command is not seen quickly enough.
        """
        await asyncio.sleep(self.max_tracking_interval)
        await self.fault(
            code=2, report=f"trackTarget not seen in {self.max_tracking_interval} sec"
        )

    async def disable_all_drives(self):
        """Stop all drives, disable them and put on brakes."""
        self._tracking_enabled = False
        already_stopped = True
        tai = utils.current_tai()
        for axis in Axis:
            actuator = self.actuators[axis]
            if actuator.kind(tai) == actuator.Kind.Stopped:
                self._axis_enabled[axis] = False
            else:
                already_stopped = False
                actuator.stop()
        self._disable_all_drives_task.cancel()
        if not already_stopped:
            self._disable_all_drives_task = asyncio.ensure_future(
                self._finish_disable_all_drives()
            )
        await self.update_events()

    async def _finish_disable_all_drives(self):
        """Wait for the main axes to stop."""
        end_times = [actuator.path[-1].tai for actuator in self.actuators]
        max_end_time = max(end_times)
        # give a bit of margin to be sure the axes are stopped
        dt = 0.1 + max_end_time - utils.current_tai()
        if dt > 0:
            await asyncio.sleep(dt)
        for axis in Axis:
            self._axis_enabled[axis] = False
        asyncio.ensure_future(self._run_update_events())

    async def _finish_stop_tracking(self):
        """Wait for the main axes to stop."""
        end_times = [self.actuators[axis].path[-1].tai for axis in MainAxes]
        max_end_time = max(end_times)
        dt = 0.1 + max_end_time - utils.current_tai()
        if dt > 0:
            await asyncio.sleep(dt)
        asyncio.ensure_future(self._run_update_events())

    async def _run_update_events(self):
        """Sleep then run update_events.

        Used to call update_events shortly after the _disable_all_drives_task
        _stop_tracking_task are done.
        """
        await asyncio.sleep(0)
        await self.update_events()

    def m3_port_rot(self, tai):
        """Return exit port and rotator axis.

        Parameters
        ----------
        tai : `float`
            Current time, TAI unix seconds.

        Returns
        -------
        port_rot : `tuple`
            Exit port and rotator axis, as a tuple:

            * exit port: an M3ExitPort enum value
            * rotator axis: the instrument rotator at this port,
              as an Axis enum value, or None if the port has no rotator.
        """
        if not self.m3_in_position(tai):
            return (None, None)
        target_position = self.actuators[Axis.M3].target.position
        for exit_port, (ind, _, rot_axis) in self._port_info_dict.items():
            if self.m3_port_positions[ind] == target_position:
                return (exit_port, rot_axis)
        return (None, None)

    def m3_in_position(self, tai):
        """Is the M3 actuator in position?

        Parameters
        ----------
        tai : `float`
            Current time, TAI unix seconds.
        """
        m3actuator = self.actuators[Axis.M3]
        if m3actuator.kind(tai) != m3actuator.Kind.Stopped:
            return False
        m3target_position = m3actuator.target.position
        m3current = m3actuator.path[-1].at(tai)
        m3position_difference = abs(m3target_position - m3current.position)
        return m3position_difference < self.m3tolerance

    async def handle_summary_state(self):
        if self.summary_state == salobj.State.ENABLED:
            axes_to_enable = set((Axis.Elevation, Axis.Azimuth))
            tai = utils.current_tai()
            rot_axis = self.m3_port_rot(tai)[1]
            if rot_axis is not None:
                axes_to_enable.add(rot_axis)
            for axis in Axis:
                self._axis_enabled[axis] = axis in axes_to_enable
        else:
            await self.disable_all_drives()
        if self.summary_state in (salobj.State.DISABLED, salobj.State.ENABLED):
            if self._events_and_telemetry_task.done():
                self._events_and_telemetry_task = asyncio.ensure_future(
                    self.events_and_telemetry_loop()
                )
        else:
            self._events_and_telemetry_task.cancel()

    async def set_write_event(self, evt_name, **kwargs):
        """Call await ``ControllerEvent.set_write`` for an event
        specified by name.

        Parameters
        ----------
        evt_name : `str`
            Event name (without the ``evt_`` prefix)
        **kwargs : `dict`
            Data for ``ControllerEvent.set``

        Returns
        -------
        result : `salobj.topics.SetWriteResult`
            Result of the call to set_write.
        """
        evt = getattr(self, f"evt_{evt_name}")
        return await evt.set_write(**kwargs)

    async def update_events(self):
        """Update most events and output those that have changed.

        Notes
        -----
        Updates state of all non-generic events, except as noted:

        * ``atMountState``
        * ``m3State``
        * (``m3PortSelected`` is output by ``do_setInstrumentPort``)

        * ``elevationInPosition``
        * ``azimuthInPosition``
        * ``nasmyth1RotatorInPosition``
        * ``nasmyth2RotatorInPosition``
        * ``m3InPosition``
        * ``allAxesInPosition``

        * ``azimuthToppleBlockCCW``
        * ``azimuthToppleBlockCW``
        * ``azimuthLimitSwitchCW``
        * ``m3RotatorDetentLimitSwitch``

        * ``elevationLimitSwitchLower``
        * ``elevationLimitSwitchUpper``
        * ``azimuthLimitSwitchCCW``
        * ``nasmyth1LimitSwitchCCW``
        * ``nasmyth1LimitSwitchCW``
        * ``nasmyth2LimitSwitchCCW``
        * ``nasmyth2LimitSwitchCW``
        * ``m3RotatorLimitSwitchCCW``
        * ``m3RotatorLimitSwitchCW``

        * ``azimuthDrive1Status``
        * ``azimuthDrive2Status``
        * ``elevationDriveStatus``
        * ``nasmyth1DriveStatus``
        * ``nasmyth2DriveStatus``
        * ``m3DriveStatus``

        * ``elevationBrake``
        * ``azimuthBrake1``
        * ``azimuthBrake2``
        * ``nasmyth1Brake``
        * ``nasmyth2Brake``

        Report events that have changed,
        and for axes that have run into a limit switch, abort the axis,
        disable its drives and set its brakes.
        """
        try:
            tai = utils.current_tai()
            current_position = np.array(
                [actuator.path.at(tai).position for actuator in self.actuators],
                dtype=float,
            )
            m3actuator = self.actuators[Axis.M3]
            axes_in_use = set([Axis.Elevation, Axis.Azimuth, Axis.M3])

            # Handle M3 actuator; set_target needs to be called to transition
            # from slewing to tracking, and that is done here for M3
            # (the trackPosition command does that for the other axes).
            m3arrived = (
                m3actuator.kind(tai) == m3actuator.Kind.Slewing
                and tai > m3actuator.path[-1].tai
            )
            if m3arrived:
                segment = simactuators.path.PathSegment(
                    tai=tai, position=m3actuator.target.position
                )
                m3actuator.path = simactuators.path.Path(
                    segment, kind=m3actuator.Kind.Stopped
                )
            exit_port, rot_axis = self.m3_port_rot(tai)
            if rot_axis is not None:
                axes_in_use.add(rot_axis)
                if m3arrived:
                    self._axis_enabled[rot_axis] = True

            # Handle limit switches
            # including aborting axes that are out of limits
            # and putting on their brakes (if any)
            abort_axes = []
            for axis in Axis:
                await self.set_write_event(
                    self._min_lim_names[axis],
                    active=current_position[axis]
                    < self.min_limit_switch_position[axis],
                )
                await self.set_write_event(
                    self._max_lim_names[axis],
                    active=current_position[axis]
                    > self.max_limit_switch_position[axis],
                )
                if (
                    current_position[axis] < self.min_limit_switch_position[axis]
                    or current_position[axis] > self.max_limit_switch_position[axis]
                ):
                    abort_axes.append(axis)
            for axis in abort_axes:
                position = current_position[axis]
                position = max(
                    position,
                    self.min_limit_switch_position[axis] - self.limit_overtravel,
                )
                position = min(
                    position,
                    self.max_limit_switch_position[axis] + self.limit_overtravel,
                )
                self.actuators[axis].abort(tai=tai, position=position)
                self._axis_enabled[axis] = False

            # Handle brakes
            for axis in Axis:
                for brake_name in self._brake_names[axis]:
                    await self.set_write_event(
                        brake_name, engaged=not self._axis_enabled[axis]
                    )

            # Handle drive status (which means enabled)
            for axis in Axis:
                for evt_name in self._drive_status_names[axis]:
                    await self.set_write_event(
                        evt_name, enable=self._axis_enabled[axis]
                    )

            # Handle atMountState
            if self._tracking_enabled:
                mount_state = AtMountState.TRACKINGENABLED
            elif (
                not self._stop_tracking_task.done()
                or not self._disable_all_drives_task.done()
            ):
                mount_state = AtMountState.STOPPING
            else:
                mount_state = AtMountState.TRACKINGDISABLED
            await self.evt_atMountState.set_write(state=mount_state)

            # Handle azimuth topple block
            if current_position[Axis.Azimuth] < self.topple_azimuth[0]:
                await self.evt_azimuthToppleBlockCCW.set_write(active=True)
                await self.evt_azimuthToppleBlockCW.set_write(active=False)
            elif current_position[Axis.Azimuth] > self.topple_azimuth[1]:
                await self.evt_azimuthToppleBlockCCW.set_write(active=False)
                await self.evt_azimuthToppleBlockCW.set_write(active=True)
            else:
                await self.evt_azimuthToppleBlockCCW.set_write(active=False)
                await self.evt_azimuthToppleBlockCW.set_write(active=False)

            # Handle m3InPosition
            # M3 is in position if the current velocity is 0
            # and the current position equals the commanded position.
            m3_in_position = self.m3_in_position(tai)
            await self.evt_m3InPosition.set_write(inPosition=m3_in_position)

            # Handle "in position" events for the main axes.
            # Main axes are in position if enabled
            # and actuator.kind(tai) is tracking.
            if not self._tracking_enabled:
                for axis in MainAxes:
                    await self.set_write_event(
                        self._in_position_names[axis], inPosition=False
                    )
                await self.evt_allAxesInPosition.set_write(inPosition=False)
            else:
                all_in_position = m3_in_position
                for axis in MainAxes:
                    if not self._axis_enabled[axis]:
                        in_position = False
                    else:
                        actuator = self.actuators[axis]
                        in_position = actuator.kind(tai) == actuator.Kind.Tracking
                    if not in_position and axis in axes_in_use:
                        all_in_position = False
                    await self.set_write_event(
                        self._in_position_names[axis], inPosition=in_position
                    )
                await self.evt_allAxesInPosition.set_write(inPosition=all_in_position)

            # compute m3_state for use setting m3State.state
            # and m3RotatorDetentSwitches
            m3_state = None
            if m3_in_position:
                # We are either at a port or at an unknown position. Use the
                # _port_info_dict to map between exit port value and m3 state.
                # If port is not mapped use unknown position. This is needed
                # in case the exit port value does not match the m3 state
                # value.
                # TODO: DM-36825 Remove mapping once the enumeration values
                # match.
                m3_state = self._port_info_dict.get(
                    exit_port, (None, M3State.UNKNOWNPOSITION, None)
                )[1]
            elif m3actuator.kind(tai) == m3actuator.Kind.Slewing:
                m3_state = M3State.INMOTION
            else:
                m3_state = M3State.UNKNOWNPOSITION
            assert m3_state is not None

            # handle m3State
            await self.evt_m3State.set_write(state=m3_state)

            # Handle M3 detent switch
            detent_map = {
                1: "nasmyth1Active",
                2: "nasmyth2Active",
                3: "port3Active",
            }
            at_field = detent_map.get(m3_state, None)
            detent_values = dict(
                (field_name, field_name == at_field)
                for field_name in detent_map.values()
            )
            await self.evt_m3RotatorDetentSwitches.set_write(**detent_values)
        except Exception as e:
            print(f"update_events failed: {e}")
            raise

    async def update_telemetry(self):
        """Output all telemetry topics."""
        try:
            nitems = len(self.tel_mount_AzEl_Encoders.data.elevationEncoder1Raw)
            curr_time = utils.current_tai()

            times = np.linspace(
                start=curr_time - self._telemetry_interval,
                stop=curr_time,
                num=nitems,
                endpoint=True,
            )

            for i, tai in enumerate(times):
                segments = [actuator.path.at(tai) for actuator in self.actuators]
                current_position = np.array(
                    [segment.position for segment in segments], dtype=float
                )
                curr_vel = np.array(
                    [segment.velocity for segment in segments], dtype=float
                )
                curr_accel = np.array(
                    [segment.acceleration for segment in segments], dtype=float
                )

                axis_encoder_counts = (
                    current_position * self.axis_encoder_counts_per_deg
                ).astype(int)
                torque = curr_accel * self.torque_per_accel
                motor_pos = current_position * self.motor_axis_ratio
                motor_pos = (motor_pos + 360) % 360 - 360
                motor_encoder_counts = (
                    motor_pos * self.motor_encoder_counts_per_deg
                ).astype(int)

                trajectory_data = self.tel_trajectory.data
                trajectory_data.elevation[i] = current_position[Axis.Elevation]
                trajectory_data.azimuth[i] = current_position[Axis.Azimuth]
                trajectory_data.nasmyth1RotatorAngle[i] = current_position[Axis.NA1]
                trajectory_data.nasmyth2RotatorAngle[i] = current_position[Axis.NA2]
                trajectory_data.elevationVelocity[i] = curr_vel[Axis.Elevation]
                trajectory_data.azimuthVelocity[i] = curr_vel[Axis.Azimuth]
                trajectory_data.nasmyth1RotatorAngleVelocity[i] = curr_vel[Axis.NA1]
                trajectory_data.nasmyth2RotatorAngleVelocity[i] = curr_vel[Axis.NA2]

                azel_encoders_data = self.tel_mount_AzEl_Encoders.data
                azel_encoders_data.elevationCalculatedAngle[i] = current_position[
                    Axis.Elevation
                ]
                azel_encoders_data.elevationEncoder1Raw[i] = axis_encoder_counts[
                    Axis.Elevation
                ]
                azel_encoders_data.elevationEncoder2Raw[i] = axis_encoder_counts[
                    Axis.Elevation
                ]
                azel_encoders_data.elevationEncoder3Raw[i] = axis_encoder_counts[
                    Axis.Elevation
                ]
                azel_encoders_data.azimuthCalculatedAngle[i] = current_position[
                    Axis.Azimuth
                ]
                azel_encoders_data.azimuthEncoder1Raw[i] = axis_encoder_counts[
                    Axis.Azimuth
                ]
                azel_encoders_data.azimuthEncoder2Raw[i] = axis_encoder_counts[
                    Axis.Azimuth
                ]
                azel_encoders_data.azimuthEncoder3Raw[i] = axis_encoder_counts[
                    Axis.Azimuth
                ]

                nasmyth_encoders_data = self.tel_mount_Nasmyth_Encoders.data
                nasmyth_encoders_data.nasmyth1CalculatedAngle[i] = current_position[
                    Axis.NA1
                ]
                nasmyth_encoders_data.nasmyth1Encoder1Raw[i] = axis_encoder_counts[
                    Axis.NA1
                ]
                nasmyth_encoders_data.nasmyth1Encoder2Raw[i] = axis_encoder_counts[
                    Axis.NA1
                ]
                nasmyth_encoders_data.nasmyth1Encoder3Raw[i] = axis_encoder_counts[
                    Axis.NA1
                ]
                nasmyth_encoders_data.nasmyth2CalculatedAngle[i] = current_position[
                    Axis.NA2
                ]
                nasmyth_encoders_data.nasmyth2Encoder1Raw[i] = axis_encoder_counts[
                    Axis.NA2
                ]
                nasmyth_encoders_data.nasmyth2Encoder2Raw[i] = axis_encoder_counts[
                    Axis.NA2
                ]
                nasmyth_encoders_data.nasmyth2Encoder3Raw[i] = axis_encoder_counts[
                    Axis.NA2
                ]

                torqueDemand_data = self.tel_torqueDemand.data
                torqueDemand_data.elevationMotorTorque[i] = torque[Axis.Elevation]
                torqueDemand_data.azimuthMotor1Torque[i] = torque[Axis.Azimuth]
                torqueDemand_data.azimuthMotor2Torque[i] = torque[Axis.Azimuth]
                torqueDemand_data.nasmyth1MotorTorque[i] = torque[Axis.NA1]
                torqueDemand_data.nasmyth2MotorTorque[i] = torque[Axis.NA2]

                measuredTorque_data = self.tel_measuredTorque.data
                measuredTorque_data.elevationMotorTorque[i] = torque[Axis.Elevation]
                measuredTorque_data.azimuthMotor1Torque[i] = torque[Axis.Azimuth]
                measuredTorque_data.azimuthMotor2Torque[i] = torque[Axis.Azimuth]
                measuredTorque_data.nasmyth1MotorTorque[i] = torque[Axis.NA1]
                measuredTorque_data.nasmyth2MotorTorque[i] = torque[Axis.NA2]

                measuredMotorVelocity_data = self.tel_measuredMotorVelocity.data
                measuredMotorVelocity_data.elevationMotorVelocity[i] = curr_vel[
                    Axis.Elevation
                ]
                measuredMotorVelocity_data.azimuthMotor1Velocity[i] = curr_vel[
                    Axis.Azimuth
                ]
                measuredMotorVelocity_data.azimuthMotor2Velocity[i] = curr_vel[
                    Axis.Azimuth
                ]
                measuredMotorVelocity_data.nasmyth1MotorVelocity[i] = curr_vel[Axis.NA1]
                measuredMotorVelocity_data.nasmyth2MotorVelocity[i] = curr_vel[Axis.NA2]

                azel_mountMotorEncoders_data = self.tel_azEl_mountMotorEncoders.data
                azel_mountMotorEncoders_data.elevationEncoder[i] = motor_pos[
                    Axis.Elevation
                ]
                azel_mountMotorEncoders_data.azimuth1Encoder[i] = motor_pos[
                    Axis.Azimuth
                ]
                azel_mountMotorEncoders_data.azimuth2Encoder[i] = motor_pos[
                    Axis.Azimuth
                ]
                azel_mountMotorEncoders_data.elevationEncoderRaw[
                    i
                ] = motor_encoder_counts[Axis.Elevation]
                azel_mountMotorEncoders_data.azimuth1EncoderRaw[
                    i
                ] = motor_encoder_counts[Axis.Azimuth]
                azel_mountMotorEncoders_data.azimuth2EncoderRaw[
                    i
                ] = motor_encoder_counts[Axis.Azimuth]

                nasmyth_m3_mountMotorEncoders_data = (
                    self.tel_nasmyth_m3_mountMotorEncoders.data
                )
                nasmyth_m3_mountMotorEncoders_data.nasmyth1Encoder[i] = motor_pos[
                    Axis.NA1
                ]
                nasmyth_m3_mountMotorEncoders_data.nasmyth2Encoder[i] = motor_pos[
                    Axis.NA2
                ]
                nasmyth_m3_mountMotorEncoders_data.m3Encoder[i] = motor_pos[Axis.M3]
                nasmyth_m3_mountMotorEncoders_data.nasmyth1EncoderRaw[
                    i
                ] = motor_encoder_counts[Axis.NA1]
                nasmyth_m3_mountMotorEncoders_data.nasmyth2EncoderRaw[
                    i
                ] = motor_encoder_counts[Axis.NA2]
                nasmyth_m3_mountMotorEncoders_data.m3EncoderRaw[
                    i
                ] = motor_encoder_counts[Axis.M3]

            await self.tel_trajectory.set_write(cRIO_timestamp=times[0])
            await self.tel_mount_AzEl_Encoders.set_write(cRIO_timestamp=times[0])
            await self.tel_mount_Nasmyth_Encoders.set_write(cRIO_timestamp=times[0])
            await self.tel_torqueDemand.set_write(cRIO_timestamp=times[0])
            await self.tel_measuredTorque.set_write(cRIO_timestamp=times[0])
            await self.tel_measuredMotorVelocity.set_write(cRIO_timestamp=times[0])
            await self.tel_azEl_mountMotorEncoders.set_write(cRIO_timestamp=times[0])
            await self.tel_nasmyth_m3_mountMotorEncoders.set_write(
                cRIO_timestamp=times[0]
            )
        except Exception as e:
            print(f"update_telemetry failed: {e}")
            raise

    async def events_and_telemetry_loop(self):
        """Output telemetry and events that have changed

        Notes
        -----
        Here are the telemetry topics that are output:

        * mountEncoders
        * torqueDemand
        * measuredTorque
        * measuredMotorVelocity
        * mountMotorEncoders

        See `update_events` for the events that are output.
        """
        i = 0
        while self.summary_state in (salobj.State.DISABLED, salobj.State.ENABLED):
            # update events first so that limits are handled
            i += 1
            await self.update_events()

            if i >= self._events_per_telemetry:
                i = 0
                await self.update_telemetry()

            await asyncio.sleep(self._telemetry_interval / self._events_per_telemetry)


def run_atmcs_simulator():
    """Run the ATMCS CSC simulator."""
    asyncio.run(ATMCSCsc.amain(index=None))
