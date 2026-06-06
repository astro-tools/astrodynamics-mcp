"""Tests for the attachment output channel.

Covers the two content-block builders and the result assembler in
:mod:`astrodynamics_mcp.attachments`, then drives one end-to-end through a
registered tool to prove the assembled ``CallToolResult`` survives FastMCP's
tool-call path with both its structured summary and its attachments intact —
the exact shape the visualisation tool bodies adopt.
"""

from __future__ import annotations

import base64

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import BaseModel, ConfigDict, Field

from astrodynamics_mcp.attachments import (
    czml_embedded_resource,
    png_image_content,
    tool_result_with_attachments,
)
from astrodynamics_mcp.server import register_tool
from astrodynamics_mcp.units import Quantity

# A tiny but non-trivial PNG byte payload — the deterministic renderer's output
# is opaque to this layer, so any bytes exercise the base64 round-trip.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
_CZML_DOC = '[{"id":"document","name":"trajectory","version":"1.0"}]'


class TestPngImageContent:
    def test_wraps_bytes_as_base64_png(self) -> None:
        content = png_image_content(_PNG_BYTES)
        assert isinstance(content, ImageContent)
        assert content.type == "image"
        assert content.mimeType == "image/png"
        # The data field is base64 and decodes back to the original bytes.
        assert base64.b64decode(content.data) == _PNG_BYTES

    def test_empty_bytes_round_trip(self) -> None:
        content = png_image_content(b"")
        assert base64.b64decode(content.data) == b""


class TestCzmlEmbeddedResource:
    def test_wraps_czml_as_text_resource(self) -> None:
        resource = czml_embedded_resource(_CZML_DOC, uri="czml://trajectory/iss")
        assert isinstance(resource, EmbeddedResource)
        assert resource.type == "resource"
        inner = resource.resource
        assert isinstance(inner, TextResourceContents)
        assert inner.text == _CZML_DOC
        assert inner.mimeType == "application/json"
        assert str(inner.uri) == "czml://trajectory/iss"


class _DummySummary(BaseModel):
    """Stand-in for a viz tool's structured summary model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    caption: str = Field(..., description="What the attachment shows.")
    time_span: Quantity = Field(..., description="Time span the plot covers.")


class TestToolResultWithAttachments:
    def test_assembles_summary_then_attachments(self) -> None:
        summary_model = _DummySummary(
            caption="ISS ground track", time_span=Quantity(value=24.0, unit="hours")
        )
        png = png_image_content(_PNG_BYTES)
        result = tool_result_with_attachments(
            structured=summary_model,
            summary="ISS ground track over 24 hours",
            attachments=[png],
        )
        assert isinstance(result, CallToolResult)
        assert result.isError is False
        # The ASCII summary leads the content list; attachments follow.
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "ISS ground track over 24 hours"
        assert result.content[1] is png
        # structuredContent mirrors a plain model return (by_alias json dump).
        assert result.structuredContent == summary_model.model_dump(mode="json", by_alias=True)

    def test_carries_multiple_attachment_kinds(self) -> None:
        summary_model = _DummySummary(caption="orbit", time_span=Quantity(value=90.0, unit="min"))
        attachments: list[ContentBlock] = [
            png_image_content(_PNG_BYTES),
            czml_embedded_resource(_CZML_DOC, uri="czml://trajectory/x"),
        ]
        result = tool_result_with_attachments(
            structured=summary_model, summary="orbit", attachments=attachments
        )
        kinds = [type(block) for block in result.content]
        assert kinds == [TextContent, ImageContent, EmbeddedResource]


class TestEndToEndThroughRegisteredTool:
    """A registered tool returning the assembled result survives the call path.

    This is the shape the visualisation tool bodies take: the function declares
    its structured summary model as the return type (so FastMCP derives an
    output schema and validates ``structuredContent`` against it), while the
    body actually returns the attachment-bearing ``CallToolResult``. FastMCP's
    convert step passes that result through verbatim, so both channels reach the
    wire together.
    """

    @pytest.fixture
    def _isolated_mcp(self, monkeypatch: pytest.MonkeyPatch) -> FastMCP:
        fresh = FastMCP("attachment-passthrough-test")
        monkeypatch.setattr("astrodynamics_mcp.server.mcp", fresh)
        return fresh

    async def test_structured_summary_and_attachments_both_reach_the_wire(
        self, _isolated_mcp: FastMCP
    ) -> None:
        png = png_image_content(_PNG_BYTES)
        czml = czml_embedded_resource(_CZML_DOC, uri="czml://trajectory/iss")

        @register_tool(
            name="dummy_viz", description="A dummy attachment-bearing tool, e.g. a plot."
        )
        async def dummy_viz() -> _DummySummary:
            model = _DummySummary(
                caption="ISS ground track", time_span=Quantity(value=24.0, unit="hours")
            )
            # The runtime object is the attachment-bearing CallToolResult; FastMCP
            # validates its structuredContent against the declared _DummySummary.
            return tool_result_with_attachments(  # type: ignore[return-value]
                structured=model,
                summary="ISS ground track over 24 hours",
                attachments=[png, czml],
            )

        result = await _isolated_mcp.call_tool("dummy_viz", {})

        assert isinstance(result, CallToolResult)
        assert result.isError is False
        assert result.structuredContent == {
            "caption": "ISS ground track",
            "time_span": {"value": 24.0, "unit": "hours"},
        }
        # Summary text leads; both attachment kinds ride alongside.
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "ISS ground track over 24 hours"
        assert any(isinstance(b, ImageContent) for b in result.content)
        assert any(isinstance(b, EmbeddedResource) for b in result.content)

    async def test_tool_exposes_an_output_schema(self, _isolated_mcp: FastMCP) -> None:
        @register_tool(
            name="dummy_viz2", description="Another dummy attachment tool, e.g. a chart."
        )
        async def dummy_viz2() -> _DummySummary:
            model = _DummySummary(caption="x", time_span=Quantity(value=1.0, unit="s"))
            return tool_result_with_attachments(  # type: ignore[return-value]
                structured=model, summary="x", attachments=[]
            )

        tools = {t.name: t for t in await _isolated_mcp.list_tools()}
        assert tools["dummy_viz2"].outputSchema is not None
