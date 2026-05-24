"""Console entry point — `astrodynamics-mcp stdio` and `astrodynamics-mcp http`.

The v0.1 skeleton wires `--help` and the two subcommand stubs. Real transport
selection (calling `mcp.run(transport=...)` on the server instance) lands in
issue #8.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from astrodynamics_mcp import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrodynamics-mcp",
        description=(
            "Model Context Protocol server for astrodynamics tools. "
            "Select a transport with the stdio or http subcommand."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="transport", metavar="{stdio,http}")

    stdio = sub.add_parser(
        "stdio",
        help="Serve over stdio (default for Claude Code, Cursor, ChatGPT desktop).",
    )
    stdio.set_defaults(func=_stdio_not_implemented)

    http = sub.add_parser(
        "http",
        help="Serve over Streamable HTTP (MCP spec 2025-11-25).",
    )
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8000)
    http.set_defaults(func=_http_not_implemented)

    return parser


def _stdio_not_implemented(_args: argparse.Namespace) -> int:
    print("astrodynamics-mcp stdio: not yet implemented (lands in issue #8).", file=sys.stderr)
    return 2


def _http_not_implemented(_args: argparse.Namespace) -> int:
    print("astrodynamics-mcp http: not yet implemented (lands in issue #8).", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.transport is None:
        parser.print_help()
        return 0
    func: object = getattr(args, "func", None)
    if callable(func):
        return int(func(args))
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
