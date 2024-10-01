# This file is part of ts_atmcssimulator.
#
# # Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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

__all__ = ["ATMCSCsc", "run_atmcs_simulator"]

import asyncio
import pathlib

from lsst.ts import attcpip, salobj

from . import __version__
from .config_schema import CONFIG_SCHEMA
from .enums import Command
from .mcs_simulator import McsSimulator


class ATMCSCsc(attcpip.AtTcpipCsc):
    """Simulator for auxiliary telescope motor control system CSC.

    Parameters
    ----------
    initial_state : `salobj.State` or `int` (optional)
        The initial state of the CSC. This is provided for unit testing,
        as real CSCs should start up in `State.STANDBY`, the default.
    """

    # TODO DM-39357 Remove these lines.
    # Append "-sim" to avoid confusion with the real ATMCS CSC.
    version = f"{__version__}-sim"

    def __init__(
        self,
        config_dir: str | pathlib.Path | None = None,
        check_if_duplicate: bool = False,
        initial_state: salobj.State = salobj.State.STANDBY,
        override: str = "",
        simulation_mode: int = 0,
    ) -> None:
        super().__init__(
            name="ATMCS",
            index=0,
            config_schema=CONFIG_SCHEMA,
            config_dir=config_dir,
            initial_state=initial_state,
            override=override,
            simulation_mode=simulation_mode,
        )

        # McsSimulator for simulation_mode == 1.
        self.simulator: McsSimulator | None = None

    async def start_clients(self) -> None:
        if self.simulation_mode == 1 and self.simulator is None:
            self.simulator = McsSimulator(
                host=self.config.host,
                cmd_evt_port=self.config.cmd_evt_port,
                telemetry_port=self.config.telemetry_port,
            )
        await super().start_clients()

    async def do_startTracking(self, data: salobj.BaseMsgType) -> None:
        self.assert_enabled("startTracking")
        command_issued = await self.write_command(command=Command.START_TRACKING)
        await command_issued.done

    async def do_trackTarget(self, data: salobj.BaseMsgType) -> None:
        self.assert_enabled("trackTarget")
        # command_issued = await self.write_command(
        await self.write_command(
            command=Command.TRACK_TARGET,
            azimuth=data.azimuth,
            azimuthVelocity=data.azimuthVelocity,
            elevation=data.elevation,
            elevationVelocity=data.elevationVelocity,
            nasmyth1RotatorAngle=data.nasmyth1RotatorAngle,
            nasmyth1RotatorAngleVelocity=data.nasmyth1RotatorAngleVelocity,
            nasmyth2RotatorAngle=data.nasmyth2RotatorAngle,
            nasmyth2RotatorAngleVelocity=data.nasmyth2RotatorAngleVelocity,
            taiTime=data.taiTime,
            trackId=data.trackId,
            tracksys=data.tracksys,
            radesys=data.radesys,
        )
        # await command_issued.done

    async def do_setInstrumentPort(self, data: salobj.BaseMsgType) -> None:
        self.assert_enabled("setInstrumentPort")
        port = data.port
        command_issued = await self.write_command(
            command=Command.SET_INSTRUMENT_PORT, port=port
        )
        await command_issued.done

    async def do_stopTracking(self, data: salobj.BaseMsgType) -> None:
        self.assert_enabled("stopTracking")
        command_issued = await self.write_command(command=Command.STOP_TRACKING)
        await command_issued.done


def run_atmcs_simulator() -> None:
    """Run the ATMCS CSC simulator."""
    asyncio.run(ATMCSCsc.amain(index=None))
