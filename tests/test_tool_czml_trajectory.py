"""Tests for the czml_trajectory tool (the gmat-czml 3D-view wrapper).

Split into two layers, like test_tool_viz_plots.py:

- The pure helpers (frame mapping, style resolution, epoch normalisation, the
  velocity-consistency guard) touch neither gmat-czml nor pandas, so they run in
  the standard test environment and carry the real input-contract coverage.
- The end-to-end tool body wraps gmat-czml's ``to_czml`` (and needs pandas, its
  dependency), both of which ship only with the ``[viz]`` extra, so that block
  self-skips where gmat-czml is absent. It is exercised in CI's ``[viz]``
  extra-install job.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, EmbeddedResource, TextContent

from astrodynamics_mcp.errors import InvalidInputError
from astrodynamics_mcp.schemas.base import Frame
from astrodynamics_mcp.tools import viz
from astrodynamics_mcp.tools.viz import (
    ContactInput,
    ContactStationInput,
    ContactWindowInput,
    CzmlTrajectoryResponse,
    CzmlTrajectoryState,
)
from astrodynamics_mcp.units import Quantity, QuantityVector


def _state(
    *,
    r: list[float],
    epoch: str,
    v: list[float] | None = (0.0, 7.5, 0.0),  # type: ignore[assignment]
    frame: Frame = Frame.TEME,
) -> CzmlTrajectoryState:
    """One state of the canonical series, with velocity optional."""
    return CzmlTrajectoryState(
        r=QuantityVector(value=r, unit="km"),
        v=None if v is None else QuantityVector(value=list(v), unit="km/s"),
        frame=frame,
        epoch=epoch,
    )


def _series(
    n: int = 4, *, frame: Frame = Frame.TEME, with_velocity: bool = True
) -> list[CzmlTrajectoryState]:
    """A short inclined state series, one point every 15 minutes."""
    import math

    states: list[CzmlTrajectoryState] = []
    for i in range(n):
        ang = math.radians(i * (360.0 / n))
        r = [7000.0 * math.cos(ang), 7000.0 * math.sin(ang), 0.0]
        v = [-7.5 * math.sin(ang), 7.5 * math.cos(ang), 0.0] if with_velocity else None
        hh, mm = divmod(i * 15, 60)
        states.append(_state(r=r, epoch=f"2024-01-01T{hh:02d}:{mm:02d}:00Z", v=v, frame=frame))
    return states


# ---------------------------------------------------------------------------
# Pure helpers — no gmat-czml / pandas required
# ---------------------------------------------------------------------------


class TestCzmlFrame:
    def test_supported_frames_map(self) -> None:
        assert viz._FRAME_TO_CZML[Frame.TEME] == "TEME"
        assert viz._FRAME_TO_CZML[Frame.ICRF] == "ICRF"
        assert viz._FRAME_TO_CZML[Frame.GCRS] == "GCRF"
        assert viz._FRAME_TO_CZML[Frame.ITRS] == "ITRF"

    def test_single_supported_frame_resolves(self) -> None:
        assert viz._czml_frame(_series(3, frame=Frame.ICRF)) == "ICRF"

    def test_mixed_frames_rejected(self) -> None:
        mixed = [
            _state(r=[7000.0, 0.0, 0.0], epoch="2024-01-01T00:00:00Z", frame=Frame.TEME),
            _state(r=[0.0, 7000.0, 0.0], epoch="2024-01-01T00:15:00Z", frame=Frame.ICRF),
        ]
        with pytest.raises(InvalidInputError) as exc:
            viz._czml_frame(mixed)
        assert exc.value.code == "invalid_input.mixed_frames"

    def test_unsupported_frame_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as exc:
            viz._czml_frame(_series(3, frame=Frame.CIRS))
        assert exc.value.code == "invalid_input.unsupported_frame"


class TestResolveStyleName:
    _PRESETS = ("sat-default", "sat-red", "sat-green", "sat-magenta")

    def test_default_alias_maps_to_sat_default(self) -> None:
        assert viz._resolve_style_name("default", self._PRESETS) == "sat-default"

    def test_known_preset_passes_through(self) -> None:
        assert viz._resolve_style_name("sat-red", self._PRESETS) == "sat-red"

    def test_unknown_style_rejected(self) -> None:
        with pytest.raises(InvalidInputError) as exc:
            viz._resolve_style_name("neon", self._PRESETS)
        assert exc.value.code == "invalid_input.unknown_style"


class TestNaiveUtc:
    def test_zulu_epoch(self) -> None:
        assert viz._naive_utc("2024-01-01T12:00:00Z") == datetime(2024, 1, 1, 12, 0, 0)

    def test_offset_epoch_converted_to_utc(self) -> None:
        # 12:00 at +05:00 is 07:00 UTC; the result is naive (tz stripped).
        result = viz._naive_utc("2024-01-01T12:00:00+05:00")
        assert result == datetime(2024, 1, 1, 7, 0, 0)
        assert result.tzinfo is None


class TestVelocityPresent:
    def test_full_velocity_series_is_present(self) -> None:
        assert viz._velocity_present(_series(3, with_velocity=True)) is True

    def test_position_only_series_is_absent(self) -> None:
        assert viz._velocity_present(_series(3, with_velocity=False)) is False

    def test_mixed_velocity_rejected(self) -> None:
        mixed = [
            _state(r=[7000.0, 0.0, 0.0], epoch="2024-01-01T00:00:00Z", v=[0.0, 7.5, 0.0]),
            _state(r=[0.0, 7000.0, 0.0], epoch="2024-01-01T00:15:00Z", v=None),
        ]
        with pytest.raises(InvalidInputError) as exc:
            viz._velocity_present(mixed)
        assert exc.value.code == "invalid_input.inconsistent_velocity"


# ---------------------------------------------------------------------------
# End-to-end tool body — requires gmat-czml (the [viz] extra)
#
# Gated in the fixture, not at module scope: a module-level importorskip would
# skip the pure-helper tests above too, but those need neither gmat-czml nor
# pandas and must run in the standard test job.
# ---------------------------------------------------------------------------


@pytest.fixture
def czml_mcp(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    """A fresh FastMCP with the viz slots registered, requiring gmat-czml."""
    pytest.importorskip("gmat_czml", reason="[viz] extra (gmat-czml) not installed")
    fresh = FastMCP("czml-trajectory-test")
    monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
    viz._register_viz_tools()
    return fresh


async def _call(mcp: FastMCP, name: str, args: dict[str, Any]) -> CallToolResult:
    """Call a tool and narrow the result to a CallToolResult for the assertions."""
    result = await mcp.call_tool(name, args)
    assert isinstance(result, CallToolResult)
    return result


def _czml_document(result: CallToolResult) -> list[Any]:
    """Pull the embedded CZML resource's parsed JSON (a list of packets)."""
    resources = [c for c in result.content if isinstance(c, EmbeddedResource)]
    assert len(resources) == 1, f"expected exactly one embedded resource, got {len(resources)}"
    resource = resources[0].resource
    assert hasattr(resource, "text"), "CZML must ride as a text resource"
    parsed = json.loads(resource.text)
    assert isinstance(parsed, list)
    return parsed


def _structured(result: CallToolResult) -> dict[str, Any]:
    assert result.structuredContent is not None, "expected structuredContent on the result"
    return result.structuredContent


def _dump(states: list[CzmlTrajectoryState]) -> list[dict[str, Any]]:
    return [s.model_dump(mode="json") for s in states]


class TestCzmlTrajectoryEndToEnd:
    async def test_returns_embedded_resource_and_summary(self, czml_mcp: FastMCP) -> None:
        result = await _call(czml_mcp, "czml_trajectory", {"trajectory": _dump(_series())})
        assert result.isError is False
        # The ASCII summary leads the content list.
        assert isinstance(result.content[0], TextContent)
        assert "CZML trajectory" in result.content[0].text
        # The structured summary round-trips through its model.
        structured = _structured(result)
        first = CzmlTrajectoryResponse.model_validate(structured).model_dump_json()
        second = CzmlTrajectoryResponse.model_validate_json(first).model_dump_json()
        assert first == second
        assert structured["frame"] == "TEME"
        assert structured["has_velocity"] is True
        assert structured["resource"]["object_count"] == 1

    async def test_emitted_czml_is_well_formed(self, czml_mcp: FastMCP) -> None:
        result = await _call(czml_mcp, "czml_trajectory", {"trajectory": _dump(_series())})
        packets = _czml_document(result)
        assert len(packets) >= 2
        preamble = packets[0]
        assert preamble["id"] == "document"
        assert "version" in preamble and preamble.get("name")
        assert "clock" in preamble
        # The satellite packet carries identity and a sampled position.
        satellite = packets[1]
        assert satellite["id"] == "satellite"
        assert "position" in satellite
        # The structured packet_count matches the emitted document.
        assert _structured(result)["resource"]["packet_count"] == len(packets)

    async def test_position_only_series_is_handled(self, czml_mcp: FastMCP) -> None:
        result = await _call(
            czml_mcp, "czml_trajectory", {"trajectory": _dump(_series(with_velocity=False))}
        )
        assert result.isError is False
        structured = _structured(result)
        assert structured["has_velocity"] is False
        # Still emits a real document (preamble + satellite).
        assert len(_czml_document(result)) >= 2

    async def test_style_preset_echoed(self, czml_mcp: FastMCP) -> None:
        result = await _call(
            czml_mcp, "czml_trajectory", {"trajectory": _dump(_series()), "style": "sat-red"}
        )
        assert _structured(result)["style"] == "sat-red"

    async def test_default_style_alias_resolves(self, czml_mcp: FastMCP) -> None:
        result = await _call(
            czml_mcp, "czml_trajectory", {"trajectory": _dump(_series()), "style": "default"}
        )
        assert _structured(result)["style"] == "sat-default"

    async def test_contacts_add_observer_and_link(self, czml_mcp: FastMCP) -> None:
        contact = ContactInput(
            station=ContactStationInput(
                name="madrid",
                lat=Quantity(value=40.43, unit="deg"),
                lon=Quantity(value=-4.25, unit="deg"),
                height=Quantity(value=0.8, unit="km"),
            ),
            windows=[ContactWindowInput(start="2024-01-01T00:05:00Z", end="2024-01-01T00:25:00Z")],
        )
        result = await _call(
            czml_mcp,
            "czml_trajectory",
            {"trajectory": _dump(_series()), "intervals": [contact.model_dump(mode="json")]},
        )
        assert result.isError is False
        ids = {p.get("id") for p in _czml_document(result)}
        # The observer entity and the observer -> satellite line of sight are present.
        assert "madrid" in ids
        assert "madrid-to-satellite" in ids
        assert _structured(result)["resource"]["contact_count"] == 1

    async def test_render_is_deterministic(self, czml_mcp: FastMCP) -> None:
        payload = {"trajectory": _dump(_series())}
        first = _czml_document(await _call(czml_mcp, "czml_trajectory", payload))
        second = _czml_document(await _call(czml_mcp, "czml_trajectory", payload))
        assert first == second

    async def test_empty_series_is_typed_error(self, czml_mcp: FastMCP) -> None:
        with pytest.raises(ToolError) as exc:
            await czml_mcp.call_tool("czml_trajectory", {"trajectory": []})
        assert "invalid_input.too_few_states" in str(exc.value)

    async def test_single_state_is_typed_error(self, czml_mcp: FastMCP) -> None:
        with pytest.raises(ToolError) as exc:
            await czml_mcp.call_tool("czml_trajectory", {"trajectory": _dump(_series(1))})
        assert "invalid_input.too_few_states" in str(exc.value)

    async def test_unsupported_frame_is_typed_error(self, czml_mcp: FastMCP) -> None:
        with pytest.raises(ToolError) as exc:
            await czml_mcp.call_tool(
                "czml_trajectory", {"trajectory": _dump(_series(frame=Frame.CIRS))}
            )
        assert "invalid_input.unsupported_frame" in str(exc.value)

    async def test_mixed_frames_is_typed_error(self, czml_mcp: FastMCP) -> None:
        mixed = [
            _state(r=[7000.0, 0.0, 0.0], epoch="2024-01-01T00:00:00Z", frame=Frame.TEME),
            _state(r=[0.0, 7000.0, 0.0], epoch="2024-01-01T00:15:00Z", frame=Frame.ICRF),
        ]
        with pytest.raises(ToolError) as exc:
            await czml_mcp.call_tool("czml_trajectory", {"trajectory": _dump(mixed)})
        assert "invalid_input.mixed_frames" in str(exc.value)

    async def test_unknown_style_is_typed_error(self, czml_mcp: FastMCP) -> None:
        with pytest.raises(ToolError) as exc:
            await czml_mcp.call_tool(
                "czml_trajectory", {"trajectory": _dump(_series()), "style": "neon"}
            )
        assert "invalid_input.unknown_style" in str(exc.value)
