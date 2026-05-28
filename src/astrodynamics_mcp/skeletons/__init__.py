"""Package containing vetted GMAT mission `.script` skeletons.

The `.script` files in this package are exposed as MCP resources by
:func:`astrodynamics_mcp.tools.gmat._register_gmat_resources`. The package
itself carries no code — it exists so `importlib.resources.files(__name__)`
can locate the skeleton files inside both source checkouts and installed
wheels.

Catalogue conventions
=====================

* **Spacecraft naming.** Generic skeletons name their spacecraft ``Sat``
  (the default name in all stock-sample-derived scripts that don't model
  a specific real mission). Orbit-class skeletons use the camel-cased
  ``<Class>Sat`` form: ``LEOSat``, ``GEOSat``, ``LunaSat``. Skeletons
  modelled after a real mission keep that mission's name as a
  one-spacecraft identifier (``MAVEN``, ``DMSP``). The constellation
  skeleton uses ``Sat<plane><slot>`` (``Sat11``, ``Sat12``, …).
* **Statement terminators.** Mission-sequence commands do *not* end in
  ``;`` -- the catalogue matches the stock-sample style for everything
  except resource-field assignments above ``BeginMissionSequence`` (where
  trailing semicolons are also optional but the bundled samples drop
  them).
* **Description header.** Every skeleton's first non-blank line is a
  ``% Description: <one short line>`` comment that
  :func:`_extract_description` surfaces as the resource's
  ``description`` field. Missing the header raises at module load.
* **No GUI subscribers, no plugin-only resources.** ``OrbitView``,
  ``GroundTrackPlot``, ``XYPlot``, ``OpenFramesInterface`` and any
  resource that requires a plugin outside ``gmat-run``'s default load
  (``SNOPT``, ``VF13``, ``CSALT``, ``MarsGRAM``, …) are stripped.
* **Epochs.** Most skeletons use ``01 Jan 2026 12:00:00.000`` or another
  arbitrary 2026 date. A few mission-driven skeletons keep a specific
  epoch where the geometry depends on it (Mars launch window, lunar
  swingby). Edit the epoch freely; the DC-targeted skeletons converge to
  new geometries when the initial state shifts.
"""
