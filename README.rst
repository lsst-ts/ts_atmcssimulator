#################
ts_atmcssimulator
#################

``ts_atmcssimulator`` is an LSST Telescope and Site package that provides a simulator for the auxiliary telescope motor control system (ATMCS).

`Documentation <https://ts-atmcssimulator.lsst.io>`_

The package is compatible with the `eups <https://github.com/RobertLuptonTheGood/eups>`_ package management system and ``scons`` build system.
Assuming you have the basic Vera C. Rubin LSST DM stack installed you can do the following, from within the package directory:

* ``setup -r .`` to setup the package and dependencies.
* ``scons`` to build the package and run unit tests.
* ``scons install declare`` to install the package and declare it to eups.
* ``package-docs build`` to build the documentation.
  This requires ``documenteer``; see `building single package docs <https://developer.lsst.io/stack/building-single-package-docs.html>`_ for installation instructions.

This code uses ``pre-commit`` to maintain ``black`` formatting and ``flake8`` compliance.
To enable this, run the following commands once (the first removes the previous pre-commit hook)::

    git config --unset-all core.hooksPath
    pre-commit install
