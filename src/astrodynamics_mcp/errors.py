"""Typed `AstrodynamicsMCPError` hierarchy with stable string error codes.

Every error raised inside a tool's body must be one of these subclasses (or the
root). The stable string `code` attribute is the wire-format contract LLM
consumers can match on — a fresh code prefix means a new error category;
extending a category means new dotted suffixes under an existing prefix.

The numeric JSON-RPC error code (e.g. `-32602` for "Invalid params") is added
by the FastMCP server layer when it converts these exceptions into MCP error
envelopes; the taxonomy here is provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar


class AstrodynamicsMCPError(Exception):
    """Root of the astrodynamics-mcp error hierarchy.

    Direct instances are valid for genuinely uncategorised failures, but in
    practice every raise site should pick the most specific subclass and a
    dotted code under that subclass's prefix.
    """

    CODE_PREFIX: ClassVar[str] = ""

    def __init__(
        self,
        message: str,
        code: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not code:
            raise ValueError("error code must be a non-empty string")
        prefix = type(self).CODE_PREFIX
        if prefix and not (code == prefix or code.startswith(f"{prefix}.")):
            raise ValueError(
                f"code {code!r} does not start with the {type(self).__name__} "
                f"prefix {prefix!r}.; expected {prefix!r} or {prefix}.<suffix>"
            )
        super().__init__(message)
        self.code = code
        self.data: dict[str, Any] = dict(data) if data is not None else {}

    @property
    def message(self) -> str:
        return self.args[0] if self.args else ""

    def to_mcp_error(self) -> dict[str, Any]:
        """Serialise to the dict that a transport layer puts on the wire.

        The shape is `{"code": str, "message": str, "data": dict}` — the
        same JSON-serialisable triple the MCP error envelope's `data` slot
        carries when the FastMCP server layer wraps the exception. The
        `code` field is preserved bit-equal through `json.dumps` / `loads`.
        """
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "data": dict(self.data),
        }
        # Cheap idempotence guard against a future caller mutating the dict.
        # JSON round-trip catches any non-serialisable value in `data` here
        # rather than at the transport boundary.
        json.dumps(payload)
        return payload


class InvalidInputError(AstrodynamicsMCPError):
    """Raised before any upstream library call when an argument fails validation.

    Use for: schema violations the pydantic layer didn't catch (e.g. semantic
    constraints like "epoch must include a time component"), unknown unit
    strings, out-of-range numeric inputs, unresolved named-entity inputs
    (e.g. unknown ground-station name).
    """

    CODE_PREFIX: ClassVar[str] = "invalid_input"


class UpstreamError(AstrodynamicsMCPError):
    """Wraps a failure from a vetted upstream library.

    Captures the original exception's type and message so the LLM consumer
    sees a stable error code plus enough context to decide whether to retry
    with different arguments or surface the problem to the user.
    """

    CODE_PREFIX: ClassVar[str] = "upstream"

    def __init__(
        self,
        message: str,
        code: str,
        *,
        original_exception: BaseException | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(data) if data is not None else {}
        if original_exception is not None:
            merged.setdefault("original_exception_type", type(original_exception).__name__)
            merged.setdefault("original_exception_message", str(original_exception))
        super().__init__(message, code, data=merged)
        self.original_exception = original_exception


class DataSourceError(AstrodynamicsMCPError):
    """Network or upstream-API failure (CelesTrak, JPL Horizons, IERS, …)."""

    CODE_PREFIX: ClassVar[str] = "data_source"

    def __init__(
        self,
        message: str,
        code: str,
        *,
        source: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not source:
            raise ValueError("DataSourceError requires a non-empty source name")
        merged: dict[str, Any] = dict(data) if data is not None else {}
        merged.setdefault("source", source)
        super().__init__(message, code, data=merged)
        self.source = source


class CredentialRequiredError(AstrodynamicsMCPError):
    """A tool needed an upstream credential the caller did not supply.

    Raised by :func:`astrodynamics_mcp.credentials.require_credential` when
    the caller has not configured a complete credential for the requested
    source. The dotted suffix names the source — e.g.
    ``credential_required.spacetrack`` — and the ``data`` dict carries the
    list of fields that were not satisfied, so the LLM consumer can
    surface a precise remediation without parsing prose.
    """

    CODE_PREFIX: ClassVar[str] = "credential_required"
