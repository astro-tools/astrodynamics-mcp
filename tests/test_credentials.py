"""Tests for `astrodynamics_mcp.credentials`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import InitializeRequestParams

from astrodynamics_mcp.credentials import (
    SOURCES,
    Credential,
    require_credential,
)
from astrodynamics_mcp.errors import CredentialRequiredError


def _clear_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ASTRODYNAMICS_MCP_<SOURCE>_<FIELD> for a clean baseline."""
    for spec in SOURCES.values():
        for field in spec.fields:
            monkeypatch.delenv(
                f"ASTRODYNAMICS_MCP_{spec.name.upper()}_{field.upper()}",
                raising=False,
            )


def _install_session(
    monkeypatch: pytest.MonkeyPatch,
    raw_meta: dict[str, Any] | None,
) -> None:
    """Patch `mcp.get_context()` to return a session with *raw_meta* as ``_meta``.

    Pass ``None`` for *raw_meta* to leave ``params.meta`` itself as ``None``
    (the client sent no ``_meta`` block at all); pass an empty dict for
    "_meta was sent but our namespaced key is absent".
    """
    payload: dict[str, Any] = {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.0"},
    }
    if raw_meta is not None:
        payload["_meta"] = raw_meta
    params = InitializeRequestParams.model_validate(payload)
    fake_context = SimpleNamespace(session=SimpleNamespace(client_params=params))

    from astrodynamics_mcp import server as server_module

    monkeypatch.setattr(server_module.mcp, "get_context", lambda: fake_context)


def _install_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate calling outside any MCP request (contextvar unset)."""
    from astrodynamics_mcp import server as server_module

    def _raise() -> Any:
        raise LookupError("no active request context")

    monkeypatch.setattr(server_module.mcp, "get_context", _raise)


class TestRegistry:
    def test_spacetrack_shape(self) -> None:
        spec = SOURCES["spacetrack"]
        assert isinstance(spec, Credential)
        assert spec.name == "spacetrack"
        assert spec.fields == ("username", "password")

    def test_discosweb_shape(self) -> None:
        spec = SOURCES["discosweb"]
        assert spec.name == "discosweb"
        assert spec.fields == ("token",)

    def test_earthdata_not_registered(self) -> None:
        """v0.3 deferred — must not silently register before its consumer ships."""
        assert "earthdata" not in SOURCES


class TestEnvVarPath:
    def test_spacetrack_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "alice")
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", "hunter2")
        assert require_credential("spacetrack") == {
            "username": "alice",
            "password": "hunter2",
        }

    def test_discosweb_single_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "bearer-abc")
        assert require_credential("discosweb") == {"token": "bearer-abc"}

    def test_partial_env_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "alice")
        # PASSWORD unset → entire credential is absent.
        with pytest.raises(CredentialRequiredError) as exc_info:
            require_credential("spacetrack")
        assert exc_info.value.data["missing_fields"] == ["password"]

    def test_empty_string_env_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "alice")
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", "")
        with pytest.raises(CredentialRequiredError) as exc_info:
            require_credential("spacetrack")
        assert "password" in exc_info.value.data["missing_fields"]


class TestSessionMetadataPath:
    def test_spacetrack_via_meta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_session(
            monkeypatch,
            {
                "astrodynamics_mcp/credentials": {
                    "spacetrack": {"username": "session-user", "password": "session-pass"}
                }
            },
        )
        assert require_credential("spacetrack") == {
            "username": "session-user",
            "password": "session-pass",
        }

    def test_discosweb_via_meta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_session(
            monkeypatch,
            {"astrodynamics_mcp/credentials": {"discosweb": {"token": "session-token"}}},
        )
        assert require_credential("discosweb") == {"token": "session-token"}

    def test_partial_meta_falls_through_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        # username present, password missing at meta layer → meta source absent.
        _install_session(
            monkeypatch,
            {"astrodynamics_mcp/credentials": {"spacetrack": {"username": "session-user"}}},
        )
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "env-user")
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", "env-pass")
        # Meta partial → env satisfies. Sources are atomic; meta's
        # `username` is *not* merged with env's `password`.
        assert require_credential("spacetrack") == {
            "username": "env-user",
            "password": "env-pass",
        }

    def test_empty_meta_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        # `_meta: {}` — sent but no namespaced key.
        _install_session(monkeypatch, {})
        monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "env-token")
        assert require_credential("discosweb") == {"token": "env-token"}

    def test_meta_omitted_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        # No `_meta` at all on the initialize request.
        _install_session(monkeypatch, None)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "env-token")
        assert require_credential("discosweb") == {"token": "env-token"}

    def test_non_mapping_meta_block_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        # Malformed: the namespaced key carries a string. Loader must ignore.
        _install_session(
            monkeypatch,
            {"astrodynamics_mcp/credentials": "not-a-mapping"},
        )
        monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "env-token")
        assert require_credential("discosweb") == {"token": "env-token"}

    def test_non_mapping_source_block_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        # Namespaced key OK; per-source value isn't a dict.
        _install_session(
            monkeypatch,
            {"astrodynamics_mcp/credentials": {"discosweb": "not-a-mapping"}},
        )
        monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "env-token")
        assert require_credential("discosweb") == {"token": "env-token"}


class TestPrecedence:
    def test_session_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_USERNAME", "env-user")
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", "env-pass")
        _install_session(
            monkeypatch,
            {
                "astrodynamics_mcp/credentials": {
                    "spacetrack": {"username": "session-user", "password": "session-pass"}
                }
            },
        )
        # HTTP overrides stdio in mixed scenarios.
        assert require_credential("spacetrack") == {
            "username": "session-user",
            "password": "session-pass",
        }


class TestMissingCredentials:
    def test_neither_source_raises_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        with pytest.raises(CredentialRequiredError) as exc_info:
            require_credential("spacetrack")
        err = exc_info.value
        assert err.code == "credential_required.spacetrack"
        assert err.data == {
            "source": "spacetrack",
            "missing_fields": ["username", "password"],
        }

    def test_discosweb_missing_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        with pytest.raises(CredentialRequiredError) as exc_info:
            require_credential("discosweb")
        assert exc_info.value.code == "credential_required.discosweb"
        assert exc_info.value.data == {
            "source": "discosweb",
            "missing_fields": ["token"],
        }

    def test_partial_meta_with_no_env_reports_only_truly_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If session meta has `username` but no env, only `password` is missing."""
        _clear_credential_env(monkeypatch)
        _install_session(
            monkeypatch,
            {"astrodynamics_mcp/credentials": {"spacetrack": {"username": "session-user"}}},
        )
        with pytest.raises(CredentialRequiredError) as exc_info:
            require_credential("spacetrack")
        # `username` is satisfied at the meta layer; `password` is missing
        # from both. The error envelope reports only the truly-missing field.
        assert exc_info.value.data == {
            "source": "spacetrack",
            "missing_fields": ["password"],
        }

    def test_partial_meta_satisfied_by_partial_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cross-source missing-fields accounting: meta has X, env has Y → still missing.

        Atomic-source rule: even though X is at meta and Y is at env, no
        single source has a *complete* credential, so the call still raises.
        The missing_fields list is empty because every field is satisfied
        somewhere — surfacing that gap is what nudges the operator to
        consolidate both fields into one source.
        """
        _clear_credential_env(monkeypatch)
        _install_session(
            monkeypatch,
            {"astrodynamics_mcp/credentials": {"spacetrack": {"username": "session-user"}}},
        )
        monkeypatch.setenv("ASTRODYNAMICS_MCP_SPACETRACK_PASSWORD", "env-pass")
        with pytest.raises(CredentialRequiredError) as exc_info:
            require_credential("spacetrack")
        assert exc_info.value.data == {
            "source": "spacetrack",
            "missing_fields": [],
        }

    def test_error_envelope_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        try:
            require_credential("spacetrack")
        except CredentialRequiredError as err:
            envelope = err.to_mcp_error()
            assert envelope["code"] == "credential_required.spacetrack"
            assert envelope["data"]["missing_fields"] == ["username", "password"]
        else:
            pytest.fail("CredentialRequiredError not raised")


class TestUnknownSource:
    def test_unknown_source_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        with pytest.raises(ValueError, match="unknown credentialled source"):
            require_credential("leolabs")

    def test_unknown_source_is_value_error_not_credential_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registry violations are programmer errors, not credential gaps."""
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        with pytest.raises(ValueError):
            require_credential("nonsense")


class TestNoContext:
    def test_env_path_works_when_get_context_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Helpers called outside a live request must not blow up."""
        _clear_credential_env(monkeypatch)
        _install_no_context(monkeypatch)
        monkeypatch.setenv("ASTRODYNAMICS_MCP_DISCOSWEB_TOKEN", "fallback-token")
        assert require_credential("discosweb") == {"token": "fallback-token"}
