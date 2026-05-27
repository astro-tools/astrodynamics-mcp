"""Credential passthrough for credentialled upstream data sources.

Some upstream sources behind the tool surface require per-user
credentials (Space-Track, ESA DISCOSweb, …). This module provides the
single lookup function tool bodies call to fetch those credentials:
:func:`require_credential`. It returns a field dict when the credential
is available, or raises a typed :class:`CredentialRequiredError` carrying
a stable string code (``credential_required.<source>``) when it is not.

Two sources are read, in this priority order:

1. **Session metadata** (HTTP transport): the MCP ``initialize`` request's
   ``_meta`` block, under the namespaced key
   ``astrodynamics_mcp/credentials``. The client passes the credentials
   once per session; they live in the FastMCP request context for the
   duration of that session.
2. **Environment variables** (stdio transport, or any fallback): the
   convention is ``ASTRODYNAMICS_MCP_<SOURCE>_<FIELD>`` upper-cased
   (``ASTRODYNAMICS_MCP_SPACETRACK_USERNAME`` and
   ``_PASSWORD``; ``ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN``).

When both sources carry a complete credential, session metadata wins —
this is the "HTTP overrides stdio in mixed scenarios" rule. A credential
is "complete" only if every required field for that source has a
non-empty string value; a partial set is treated as absent and falls
through to the next source.

This module never logs, prints, or otherwise leaks credential values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final, NamedTuple

from astrodynamics_mcp.errors import CredentialRequiredError

_META_NAMESPACE: Final[str] = "astrodynamics_mcp/credentials"
_ENV_PREFIX: Final[str] = "ASTRODYNAMICS_MCP_"


class Credential(NamedTuple):
    """Schema entry for one credentialled source.

    Attributes:
        name: The lowercase source identifier used everywhere on the
            wire (env-var middle segment, ``_meta`` key, error-code
            suffix). E.g. ``"spacetrack"``, ``"discosweb"``.
        fields: The required field names for this source. All must be
            present and non-empty for the credential to count as
            available.
    """

    name: str
    fields: tuple[str, ...]


SOURCES: Final[Mapping[str, Credential]] = {
    "spacetrack": Credential(name="spacetrack", fields=("username", "password")),
    "discosweb": Credential(name="discosweb", fields=("token",)),
}
"""The registry of known credentialled sources.

Each entry is referenced by its lowercase name in environment-variable
names, ``_meta`` keys, and error codes. A new credentialled source means
a new entry here and a new row in the docs matrix.
"""


def _env_var_name(source: str, field: str) -> str:
    """Compose the env-var name for ``source``/``field`` per the convention."""
    return f"{_ENV_PREFIX}{source.upper()}_{field.upper()}"


def _load_from_env(spec: Credential) -> dict[str, str] | None:
    """Read every required field for *spec* from the process environment.

    Returns the assembled dict if every field has a non-empty value;
    returns ``None`` if any field is missing or empty (partial = absent).
    """
    out: dict[str, str] = {}
    for field in spec.fields:
        value = os.environ.get(_env_var_name(spec.name, field), "")
        if not value:
            return None
        out[field] = value
    return out


def _session_meta_block() -> Mapping[str, Any] | None:
    """Return the ``astrodynamics_mcp/credentials`` block from the live session, if any.

    Looks up the active FastMCP request context, walks down to the
    initialize request's ``_meta`` field, and returns the value at our
    namespaced key. Returns ``None`` if any link in that chain is missing
    or doesn't match the expected shape — including the common case of
    being called outside an MCP request entirely (no contextvar set).
    The session-metadata path is best-effort; env vars are the fallback.
    """
    try:
        from astrodynamics_mcp.server import mcp

        context = mcp.get_context()
        session = context.session
        params = session.client_params
    except (LookupError, AttributeError, RuntimeError, ValueError):
        # LookupError: contextvar unset (no active request).
        # AttributeError: SDK shape changed or session not yet initialized.
        # RuntimeError: FastMCP raises this when there's no request context.
        # ValueError: defensive — pydantic occasionally surfaces these.
        return None
    if params is None or params.meta is None:
        return None
    # `meta` is a pydantic model with extra="allow"; extras land in
    # `model_extra`. The namespaced key is not a declared field, so it
    # always arrives via extras.
    extras = params.meta.model_extra or {}
    block = extras.get(_META_NAMESPACE)
    if not isinstance(block, Mapping):
        return None
    return block


def _load_from_session(spec: Credential) -> dict[str, str] | None:
    """Read every required field for *spec* from the live session's ``_meta``."""
    block = _session_meta_block()
    if block is None:
        return None
    source_block = block.get(spec.name)
    if not isinstance(source_block, Mapping):
        return None
    out: dict[str, str] = {}
    for field in spec.fields:
        value = source_block.get(field)
        if not isinstance(value, str) or not value:
            return None
        out[field] = value
    return out


def require_credential(source: str) -> dict[str, str]:
    """Return the credential dict for *source*, or raise if it is unavailable.

    Source-aware: session metadata (HTTP) takes precedence over environment
    variables (stdio) when both sources carry a complete credential. A
    partial credential is treated as absent at each source — every required
    field for the source must have a non-empty string value.

    Args:
        source: The lowercase source identifier, e.g. ``"spacetrack"``.
            Must be a key of :data:`SOURCES`.

    Returns:
        A dict keyed by the source's required field names, with non-empty
        string values. Tool bodies should treat the dict as opaque
        secret material — do not log it, do not echo it into a tool
        response, do not write it to disk.

    Raises:
        ValueError: If *source* is not a registered credentialled source.
        CredentialRequiredError: If neither source carries a complete
            credential. The error's ``code`` is
            ``credential_required.<source>``; its ``data`` carries
            ``{"source": "<source>", "missing_fields": [...]}`` so the
            LLM consumer can surface the gap without parsing prose.
    """
    spec = SOURCES.get(source)
    if spec is None:
        raise ValueError(
            f"unknown credentialled source {source!r}; expected one of: {sorted(SOURCES)}"
        )

    from_session = _load_from_session(spec)
    if from_session is not None:
        return from_session

    from_env = _load_from_env(spec)
    if from_env is not None:
        return from_env

    missing = _missing_fields(spec)
    raise CredentialRequiredError(
        f"no credential available for {spec.name!r}; "
        f"set the env vars or pass them via the MCP initialize _meta block",
        code=f"credential_required.{spec.name}",
        data={"source": spec.name, "missing_fields": missing},
    )


def _missing_fields(spec: Credential) -> list[str]:
    """List the fields not satisfied by either source, for the error envelope."""
    block = _session_meta_block()
    session_block: Mapping[str, Any] = {}
    if block is not None:
        candidate = block.get(spec.name)
        if isinstance(candidate, Mapping):
            session_block = candidate
    missing: list[str] = []
    for field in spec.fields:
        session_value = session_block.get(field) if session_block else None
        if isinstance(session_value, str) and session_value:
            continue
        env_value = os.environ.get(_env_var_name(spec.name, field), "")
        if env_value:
            continue
        missing.append(field)
    return missing
