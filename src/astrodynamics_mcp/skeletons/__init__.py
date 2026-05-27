"""Package containing vetted GMAT mission `.script` skeletons.

The `.script` files in this package are exposed as MCP resources by
:func:`astrodynamics_mcp.tools.gmat._register_gmat_resources`. The package
itself carries no code — it exists so `importlib.resources.files(__name__)`
can locate the skeleton files inside both source checkouts and installed
wheels.
"""
