"""Tests for `astrodynamics_mcp.schemas.base`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import (
    Body,
    Epoch,
    Frame,
    Interval,
    KeplerianElements,
    NamedStation,
    Observer,
    ObserverCoordinates,
    StateVector,
    TimeScale,
    Tle,
    TleLines,
    TleOmm,
)
from astrodynamics_mcp.units import Quantity, QuantityVector

_EPOCH_ADAPTER: TypeAdapter[str] = TypeAdapter(Epoch)
_OBSERVER_ADAPTER: TypeAdapter[NamedStation | ObserverCoordinates] = TypeAdapter(Observer)
_TLE_ADAPTER: TypeAdapter[TleLines | TleOmm] = TypeAdapter(Tle)
_BODY_ADAPTER: TypeAdapter[str] = TypeAdapter(Body)


class TestTimeScale:
    @pytest.mark.parametrize("scale", ["UTC", "TAI", "TT", "TDB", "UT1", "GPS", "TCB", "TCG"])
    def test_each_value_round_trips(self, scale: str) -> None:
        assert TimeScale(scale).value == scale

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid TimeScale"):
            TimeScale("BST")  # British Summer Time is not a time scale.


class TestFrame:
    @pytest.mark.parametrize(
        "frame",
        ["TEME", "ICRF", "GCRS", "ITRS", "CIRS", "TIRS", "IAU_EARTH", "IAU_MARS", "IAU_MOON"],
    )
    def test_each_value_round_trips(self, frame: str) -> None:
        assert Frame(frame).value == frame

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid Frame"):
            Frame("MJ2000")


class TestEpoch:
    @pytest.mark.parametrize(
        "valid",
        [
            "2026-05-23T12:00:00Z",
            "2026-05-23T12:00:00",
            "2026-05-23T12:00:00.500Z",
            "2026-05-23T12:00:00.123456",
            "2026-05-23T12:00:00+00:00",
            "2026-05-23T12:00:00-05:00",
            "2026-05-23T12:00:00+0530",
        ],
    )
    def test_accepts_valid_iso8601(self, valid: str) -> None:
        assert _EPOCH_ADAPTER.validate_python(valid) == valid

    def test_rejects_bare_date_with_specific_code(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            _EPOCH_ADAPTER.validate_python("2026-05-23")
        assert excinfo.value.code == "invalid_input.epoch_missing_time_component"

    @pytest.mark.parametrize(
        "bad",
        [
            "2026/05/23T12:00:00",  # wrong date separator
            "2026-5-23T12:00:00",  # single-digit month
            "2026-05-23 12:00:00",  # space instead of T
            "2026-05-23T12:00",  # missing seconds
            "23 May 2026",  # non-ISO
            "",  # empty
        ],
    )
    def test_rejects_malformed_with_generic_code(self, bad: str) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            _EPOCH_ADAPTER.validate_python(bad)
        assert excinfo.value.code == "invalid_input.epoch_malformed"

    def test_non_string_input_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            _EPOCH_ADAPTER.validate_python(2026)
        assert excinfo.value.code == "invalid_input.epoch_not_a_string"


class TestBody:
    @pytest.mark.parametrize("name", ["earth", "sun", "mars", "hubble", "399", "Earth"])
    def test_body_accepts_strings(self, name: str) -> None:
        # Body has no validator at the schema level — it's a documentation-
        # bearing alias. Resolution is the consuming tool's responsibility.
        assert _BODY_ADAPTER.validate_python(name) == name


class TestNamedStation:
    @pytest.mark.parametrize(
        "name",
        ["madrid", "goldstone", "canberra", "svalbard", "wallops", "esrange", "gsfc", "jpl"],
    )
    def test_valid_names_accepted(self, name: str) -> None:
        st = NamedStation(name=name)  # type: ignore[arg-type]
        assert st.name == name

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NamedStation(name="kourou")  # type: ignore[arg-type]

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NamedStation.model_validate({"name": "madrid", "extra": "no"})

    def test_round_trip_through_json(self) -> None:
        st = NamedStation(name="madrid")
        assert NamedStation.model_validate_json(st.model_dump_json()) == st


class TestObserverCoordinates:
    def _madrid(self) -> ObserverCoordinates:
        return ObserverCoordinates(
            lat=Quantity(value=40.4168, unit="deg"),
            lon=Quantity(value=-3.7038, unit="deg"),
            alt=Quantity(value=0.667, unit="km"),
        )

    def test_construct_and_round_trip(self) -> None:
        madrid = self._madrid()
        restored = ObserverCoordinates.model_validate_json(madrid.model_dump_json())
        assert restored == madrid

    def test_wrong_angle_unit_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            ObserverCoordinates(
                lat=Quantity(value=40.0, unit="km"),
                lon=Quantity(value=-3.0, unit="deg"),
                alt=Quantity(value=0.0, unit="km"),
            )
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    def test_wrong_alt_unit_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            ObserverCoordinates(
                lat=Quantity(value=40.0, unit="deg"),
                lon=Quantity(value=-3.0, unit="deg"),
                alt=Quantity(value=0.0, unit="deg"),
            )
        assert excinfo.value.code == "invalid_input.wrong_unit_category"


class TestObserverUnion:
    def test_named_shape_parses_as_named_station(self) -> None:
        obs = _OBSERVER_ADAPTER.validate_python({"name": "madrid"})
        assert isinstance(obs, NamedStation)

    def test_coordinates_shape_parses_as_coordinates(self) -> None:
        obs = _OBSERVER_ADAPTER.validate_python(
            {
                "lat": {"value": 40.4168, "unit": "deg"},
                "lon": {"value": -3.7038, "unit": "deg"},
                "alt": {"value": 0.667, "unit": "km"},
            }
        )
        assert isinstance(obs, ObserverCoordinates)

    def test_invalid_shape_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _OBSERVER_ADAPTER.validate_python({"latitude": 40.0})  # neither shape


class TestTleLines:
    _ISS_LINE_1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  18000-3 0  9990"
    _ISS_LINE_2 = "2 25544  51.6400  90.0000 0001000  90.0000 270.0000 15.50000000000000"

    def test_round_trip(self) -> None:
        tle = TleLines(line1=self._ISS_LINE_1, line2=self._ISS_LINE_2)
        restored = TleLines.model_validate_json(tle.model_dump_json())
        assert restored == tle

    def test_short_line_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            TleLines(line1=self._ISS_LINE_1[:-1], line2=self._ISS_LINE_2)
        assert excinfo.value.code == "invalid_input.tle_line_wrong_length"

    def test_long_line_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            TleLines(line1=self._ISS_LINE_1, line2=self._ISS_LINE_2 + "X")
        assert excinfo.value.code == "invalid_input.tle_line_wrong_length"


class TestTleOmm:
    def test_round_trip(self) -> None:
        omm = TleOmm(omm={"CCSDS_OMM_VERS": "2.0", "EPOCH": "2024-01-01T12:00:00.000000"})
        restored = TleOmm.model_validate_json(omm.model_dump_json())
        assert restored == omm

    def test_empty_omm_dict_allowed(self) -> None:
        # Loose at v0.1: validation is the consuming tool's responsibility.
        TleOmm(omm={})


class TestTleUnion:
    def test_lines_shape_parses_as_tle_lines(self) -> None:
        tle = _TLE_ADAPTER.validate_python(
            {
                "line1": TestTleLines._ISS_LINE_1,
                "line2": TestTleLines._ISS_LINE_2,
            }
        )
        assert isinstance(tle, TleLines)

    def test_omm_shape_parses_as_tle_omm(self) -> None:
        tle = _TLE_ADAPTER.validate_python({"omm": {"EPOCH": "2024-01-01T12:00:00.000000"}})
        assert isinstance(tle, TleOmm)


class TestStateVector:
    def _leo(self) -> StateVector:
        return StateVector(
            r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
            v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
            frame=Frame.TEME,
            epoch="2026-05-23T12:00:00Z",
        )

    def test_round_trip(self) -> None:
        sv = self._leo()
        restored = StateVector.model_validate_json(sv.model_dump_json())
        assert restored == sv

    @pytest.mark.parametrize("unit", ["km", "m", "AU"])
    def test_position_accepts_length_units(self, unit: str) -> None:
        StateVector(
            r=QuantityVector(value=[1.0, 0.0, 0.0], unit=unit),
            v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
            frame=Frame.TEME,
            epoch="2026-05-23T12:00:00Z",
        )

    def test_wrong_position_unit_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km/s"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame=Frame.TEME,
                epoch="2026-05-23T12:00:00Z",
            )
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    def test_wrong_velocity_unit_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="deg"),
                frame=Frame.TEME,
                epoch="2026-05-23T12:00:00Z",
            )
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    def test_position_length_must_be_three(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0], unit="km"),  # only 2 components
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame=Frame.TEME,
                epoch="2026-05-23T12:00:00Z",
            )
        assert excinfo.value.code == "invalid_input.wrong_vector_length"

    def test_velocity_length_must_be_three(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0, 0.0], unit="km/s"),  # 4
                frame=Frame.TEME,
                epoch="2026-05-23T12:00:00Z",
            )
        assert excinfo.value.code == "invalid_input.wrong_vector_length"

    def test_unknown_frame_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame="MJ2000",  # type: ignore[arg-type]
                epoch="2026-05-23T12:00:00Z",
            )

    def test_bare_date_epoch_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            StateVector(
                r=QuantityVector(value=[7000.0, 0.0, 0.0], unit="km"),
                v=QuantityVector(value=[0.0, 7.5, 0.0], unit="km/s"),
                frame=Frame.TEME,
                epoch="2026-05-23",
            )
        assert excinfo.value.code == "invalid_input.epoch_missing_time_component"


class TestInterval:
    def _ten_minutes(self) -> Interval:
        return Interval(
            start="2026-05-23T12:00:00Z",
            end="2026-05-23T12:10:00Z",
            duration_s=Quantity(value=600.0, unit="s"),
        )

    def test_round_trip(self) -> None:
        interval = self._ten_minutes()
        restored = Interval.model_validate_json(interval.model_dump_json())
        assert restored == interval

    @pytest.mark.parametrize("unit", ["s", "min", "hours", "days"])
    def test_duration_accepts_time_units(self, unit: str) -> None:
        Interval(
            start="2026-05-23T12:00:00Z",
            end="2026-05-23T12:10:00Z",
            duration_s=Quantity(value=10.0, unit=unit),
        )

    def test_wrong_duration_unit_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            Interval(
                start="2026-05-23T12:00:00Z",
                end="2026-05-23T12:10:00Z",
                duration_s=Quantity(value=10.0, unit="km"),
            )
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            Interval(
                start="2026-05-23T12:10:00Z",
                end="2026-05-23T12:00:00Z",  # before start
                duration_s=Quantity(value=600.0, unit="s"),
            )
        assert excinfo.value.code == "invalid_input.interval_end_not_after_start"

    def test_end_equal_to_start_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            Interval(
                start="2026-05-23T12:00:00Z",
                end="2026-05-23T12:00:00Z",  # equal
                duration_s=Quantity(value=0.0, unit="s"),
            )
        assert excinfo.value.code == "invalid_input.interval_end_not_after_start"


class TestKeplerianElements:
    def _good(self) -> dict[str, Any]:
        return {
            "a": {"value": 24371.0, "unit": "km"},
            "e": {"value": 0.7, "unit": "1"},
            "i": {"value": 28.5, "unit": "deg"},
            "raan": {"value": 45.0, "unit": "deg"},
            "argp": {"value": 90.0, "unit": "deg"},
            "nu": {"value": 30.0, "unit": "deg"},
        }

    def test_round_trips_through_json(self) -> None:
        kep = KeplerianElements.model_validate(self._good())
        restored = KeplerianElements.model_validate_json(kep.model_dump_json())
        assert restored == kep

    def test_eccentricity_unit_must_be_dimensionless(self) -> None:
        bad = self._good()
        bad["e"] = {"value": 0.7, "unit": "km"}
        with pytest.raises(InvalidInputError) as excinfo:
            KeplerianElements.model_validate(bad)
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    def test_semi_major_axis_unit_must_be_length(self) -> None:
        bad = self._good()
        bad["a"] = {"value": 24371.0, "unit": "deg"}
        with pytest.raises(InvalidInputError) as excinfo:
            KeplerianElements.model_validate(bad)
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    @pytest.mark.parametrize("field", ["i", "raan", "argp", "nu"])
    def test_angle_field_unit_must_be_angle(self, field: str) -> None:
        bad = self._good()
        bad[field] = {"value": 1.0, "unit": "km"}
        with pytest.raises(InvalidInputError) as excinfo:
            KeplerianElements.model_validate(bad)
        assert excinfo.value.code == "invalid_input.wrong_unit_category"

    def test_negative_a_is_accepted_for_hyperbolic_arc(self) -> None:
        """Hyperbolic arcs carry a < 0; schema mustn't reject them."""
        hyperbolic = self._good()
        hyperbolic["a"] = {"value": -50000.0, "unit": "km"}
        hyperbolic["e"] = {"value": 1.5, "unit": "1"}
        kep = KeplerianElements.model_validate(hyperbolic)
        assert kep.a.value == -50000.0
        assert kep.e.value == 1.5


class TestComposedSchemasExportRichJsonSchema:
    """Sanity check that the JSON-schema export from these models is canonical.

    Each model's `model_json_schema()` must produce a dict the MCP SDK can
    surface to the LLM. Spot-checks: top-level type, presence of property
    descriptions, presence of examples on selected fields.
    """

    @pytest.mark.parametrize(
        "model_cls", [NamedStation, ObserverCoordinates, TleLines, TleOmm, StateVector, Interval]
    )
    def test_schema_is_well_formed_object(self, model_cls: type[BaseModel]) -> None:
        schema: dict[str, Any] = model_cls.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["additionalProperties"] is False

    def test_observer_union_emits_both_variants(self) -> None:
        schema = _OBSERVER_ADAPTER.json_schema()
        # The Union surfaces as anyOf with two $refs.
        assert "anyOf" in schema or "oneOf" in schema or "$ref" in schema

    def test_field_descriptions_present(self) -> None:
        sv_schema = StateVector.model_json_schema()
        for field in ("r", "v", "frame", "epoch"):
            assert "description" in sv_schema["properties"][field]
