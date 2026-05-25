"""Tests for `astrodynamics_mcp.errors`."""

from __future__ import annotations

import json

import pytest

from astrodynamics_mcp.errors import (
    AstrodynamicsMCPError,
    CredentialRequiredError,
    DataSourceError,
    InvalidInputError,
    UpstreamError,
)


class TestRoot:
    def test_message_and_code_round_trip(self) -> None:
        err = AstrodynamicsMCPError("something went wrong", "freeform.code")
        assert err.message == "something went wrong"
        assert err.code == "freeform.code"
        assert str(err) == "something went wrong"

    def test_data_defaults_to_empty_dict(self) -> None:
        err = AstrodynamicsMCPError("x", "freeform")
        assert err.data == {}

    def test_data_payload_is_copied_not_aliased(self) -> None:
        payload = {"k": "v"}
        err = AstrodynamicsMCPError("x", "freeform", data=payload)
        payload["k"] = "mutated"
        assert err.data == {"k": "v"}

    def test_empty_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AstrodynamicsMCPError("x", "")

    def test_to_mcp_error_shape(self) -> None:
        err = InvalidInputError("bad", "invalid_input.foo", data={"hint": "use bar"})
        envelope = err.to_mcp_error()
        assert envelope == {
            "code": "invalid_input.foo",
            "message": "bad",
            "data": {"hint": "use bar"},
        }

    def test_to_mcp_error_round_trips_through_json(self) -> None:
        """The `code` field must survive JSON serialisation bit-equal."""
        err = InvalidInputError("bad", "invalid_input.epoch_missing_time_component")
        envelope = err.to_mcp_error()
        restored = json.loads(json.dumps(envelope))
        assert restored["code"] == err.code
        assert restored["message"] == err.message
        assert restored["data"] == err.data

    def test_to_mcp_error_returned_dict_is_an_independent_copy(self) -> None:
        err = AstrodynamicsMCPError("x", "freeform", data={"k": 1})
        envelope = err.to_mcp_error()
        envelope["data"]["k"] = 999
        assert err.data == {"k": 1}

    def test_non_serialisable_data_fails_fast(self) -> None:
        class Opaque:
            pass

        err = AstrodynamicsMCPError("x", "freeform", data={"obj": Opaque()})
        with pytest.raises(TypeError):
            err.to_mcp_error()


class TestPrefixEnforcement:
    @pytest.mark.parametrize(
        ("cls", "good_codes", "bad_codes"),
        [
            (
                InvalidInputError,
                ["invalid_input", "invalid_input.unknown_unit", "invalid_input.foo.bar"],
                ["invalid_inputs", "input.bad", "", "INVALID_INPUT.foo"],
            ),
            (
                UpstreamError,
                ["upstream", "upstream.sgp4_failure", "upstream.lamberthub.no_solution"],
                ["upstreams", "sgp4_failure", "upstream_other"],
            ),
            (
                DataSourceError,
                ["data_source", "data_source.celestrak_unreachable"],
                ["data_sources", "celestrak.unreachable"],
            ),
            (
                CredentialRequiredError,
                ["credential_required", "credential_required.space_track"],
                ["credentials_required", "credential_required_thing"],
            ),
        ],
    )
    def test_prefix(
        self,
        cls: type[AstrodynamicsMCPError],
        good_codes: list[str],
        bad_codes: list[str],
    ) -> None:
        for code in good_codes:
            err = self._construct(cls, code)
            assert err.code == code
        for code in bad_codes:
            with pytest.raises(ValueError):
                self._construct(cls, code)

    def test_root_class_has_no_prefix_constraint(self) -> None:
        AstrodynamicsMCPError("x", "anything.at.all")

    @staticmethod
    def _construct(cls: type[AstrodynamicsMCPError], code: str) -> AstrodynamicsMCPError:
        if issubclass(cls, DataSourceError):
            return cls("x", code, source="celestrak")
        return cls("x", code)


class TestUpstreamError:
    def test_captures_original_exception_type_and_message(self) -> None:
        original = ValueError("sgp4 said no")
        err = UpstreamError(
            "sgp4 propagation failed",
            "upstream.sgp4_failure",
            original_exception=original,
        )
        assert err.original_exception is original
        assert err.data["original_exception_type"] == "ValueError"
        assert err.data["original_exception_message"] == "sgp4 said no"

    def test_no_original_exception_leaves_data_alone(self) -> None:
        err = UpstreamError("x", "upstream.x")
        assert "original_exception_type" not in err.data
        assert "original_exception_message" not in err.data

    def test_caller_supplied_data_takes_precedence_over_synthesised(self) -> None:
        err = UpstreamError(
            "x",
            "upstream.x",
            original_exception=ValueError("from arg"),
            data={"original_exception_message": "from caller"},
        )
        assert err.data["original_exception_message"] == "from caller"
        assert err.data["original_exception_type"] == "ValueError"


class TestDataSourceError:
    def test_source_is_required(self) -> None:
        with pytest.raises(ValueError, match="non-empty source"):
            DataSourceError("x", "data_source.x", source="")

    def test_source_lands_in_data(self) -> None:
        err = DataSourceError(
            "celestrak fetch failed",
            "data_source.celestrak_unreachable",
            source="celestrak",
        )
        assert err.source == "celestrak"
        assert err.data["source"] == "celestrak"

    def test_caller_supplied_source_in_data_is_preserved(self) -> None:
        err = DataSourceError(
            "x",
            "data_source.x",
            source="celestrak",
            data={"source": "override"},
        )
        assert err.data["source"] == "override"


class TestCredentialRequiredError:
    def test_defined_and_constructible(self) -> None:
        err = CredentialRequiredError("Space-Track auth missing", "credential_required.space_track")
        assert isinstance(err, AstrodynamicsMCPError)
        assert err.code == "credential_required.space_track"
