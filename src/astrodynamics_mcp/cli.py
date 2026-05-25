"""Console entry point — `astrodynamics-mcp stdio` and `astrodynamics-mcp http`.

Both subcommands dispatch to the module-level :data:`astrodynamics_mcp.server.mcp`
singleton so any tool registration that happened at import time is preserved.
The CLI is just the two-entry-point glue — no tool logic lives here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from astrodynamics_mcp import __version__

_LOG_LEVELS = ("debug", "info", "warning", "error")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrodynamics-mcp",
        description=(
            "Model Context Protocol server for astrodynamics tools. "
            "Select a transport with the stdio or http subcommand."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="warning",
        help="Logging level (default: warning).",
    )

    sub = parser.add_subparsers(dest="transport", metavar="{stdio,http}")

    stdio = sub.add_parser(
        "stdio",
        help="Serve over stdio (default for Claude Code, Cursor, ChatGPT desktop).",
    )
    stdio.set_defaults(func=_run_stdio)

    http = sub.add_parser(
        "http",
        help="Serve over Streamable HTTP (MCP spec 2025-11-25).",
    )
    http.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; pass 0.0.0.0 for all interfaces).",
    )
    http.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000).",
    )
    http.set_defaults(func=_run_http)

    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.WARNING)
    # `basicConfig` is a no-op once the root logger has handlers, so we set
    # the level on the root logger explicitly (idempotent across calls). The
    # handler is only added once — `basicConfig(force=False)` short-circuits
    # on subsequent invocations.
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger().setLevel(level)


def _run_stdio(_args: argparse.Namespace) -> int:
    """Run the server over stdio. Blocks until the client disconnects."""
    from astrodynamics_mcp.server import mcp

    mcp.run(transport="stdio")
    return 0


def _run_http(args: argparse.Namespace) -> int:
    """Run the server over Streamable HTTP. Blocks until SIGINT / SIGTERM."""
    from astrodynamics_mcp.server import mcp

    # FastMCP's run() takes only (transport, mount_path); host / port live on
    # the settings object. Mutate just before launch so the singleton's
    # registered tools stay intact.
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    try:
        mcp.run(transport="streamable-http")
    except OSError as exc:
        # Port already in use, permission denied on low ports, etc. The MCP
        # SDK doesn't currently surface these as DataSourceError, but the
        # CLI does the user-friendly translation.
        print(
            f"astrodynamics-mcp http: failed to bind {args.host}:{args.port} — {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    if args.transport is None:
        parser.print_help()
        return 0
    func: object = getattr(args, "func", None)
    if not callable(func):
        parser.print_help()
        return 0
    try:
        return int(func(args))
    except KeyboardInterrupt:
        # The SIGINT exit code convention; matches what Python's default
        # KeyboardInterrupt handling would yield from a bare `raise`, but
        # we make it explicit so the CLI contract is documented.
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
