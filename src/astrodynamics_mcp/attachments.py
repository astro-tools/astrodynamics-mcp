"""Attachment output channel — structured summary plus binary/resource extras.

The numeric tool surface returns a single pydantic model, which FastMCP turns
into structured content plus a JSON text block. The visualisation tools need
more: a rendered PNG or a CZML document carried *alongside* that summary, not
instead of it. This module is the reusable plumbing for that — it builds the
content blocks and assembles the :class:`~mcp.types.CallToolResult` the viz
tool bodies return.

FastMCP's tool-call path passes a returned ``CallToolResult`` through verbatim
(validating its ``structuredContent`` against the tool's output model when one
is declared), so a tool that returns :func:`tool_result_with_attachments`
surfaces three things on the wire at once:

- ``structuredContent`` — the tool's structured summary model, the programmatic
  channel a downstream consumer reads;
- a leading ``TextContent`` — the human-/LLM-facing ASCII summary, the default
  every client renders;
- the attachment blocks — an :class:`~mcp.types.ImageContent` (a PNG) or an
  :class:`~mcp.types.EmbeddedResource` (a CZML document) a richer client can
  display.

Attachments are strictly additive: the ASCII summary is always present and is
what a text-only client sees, so no client is left with nothing to render.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence

from mcp.types import (
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl, BaseModel

# Media types for the two attachment kinds the viz tools emit. CZML is a JSON
# document (a Cesium-specific schema), so it rides as application/json text.
_PNG_MEDIA_TYPE = "image/png"
_CZML_MEDIA_TYPE = "application/json"


def png_image_content(data: bytes) -> ImageContent:
    """Wrap raw PNG bytes as an MCP :class:`~mcp.types.ImageContent` block.

    The bytes are base64-encoded onto the ``data`` field (the wire shape MCP
    image content uses); the media type is fixed to ``image/png`` because this
    is the one raster format the deterministic renderer
    (:func:`astrodynamics_mcp.viz_render.render_png`) emits.
    """
    encoded = base64.b64encode(data).decode("ascii")
    return ImageContent(type="image", data=encoded, mimeType=_PNG_MEDIA_TYPE)


def czml_embedded_resource(czml: str, *, uri: str) -> EmbeddedResource:
    """Wrap a CZML document as an MCP :class:`~mcp.types.EmbeddedResource` block.

    The CZML text rides as an ``application/json`` text resource keyed by *uri*
    (a stable identifier the client can reference, e.g.
    ``czml://trajectory/iss``). The document itself is the payload; resource
    presentation (display title, styling) is the CZML tool's concern, not this
    transport helper's.
    """
    resource = TextResourceContents(uri=AnyUrl(uri), mimeType=_CZML_MEDIA_TYPE, text=czml)
    return EmbeddedResource(type="resource", resource=resource)


def tool_result_with_attachments(
    *,
    structured: BaseModel,
    summary: str,
    attachments: Sequence[ContentBlock],
) -> CallToolResult:
    """Assemble a tool result carrying the summary, an ASCII block, and attachments.

    *structured* is the tool's response model — its JSON dump becomes
    ``structuredContent`` (and is validated against the tool's declared output
    model on the way out). *summary* is the always-present ASCII text block,
    rendered first so it leads the content list. *attachments* are the additive
    PNG / CZML blocks built by :func:`png_image_content` /
    :func:`czml_embedded_resource`.

    The ``by_alias`` dump mirrors how FastMCP serialises a plain model return,
    so a tool that adopts this channel produces the same ``structuredContent``
    it would have without attachments.
    """
    content: list[ContentBlock] = [TextContent(type="text", text=summary)]
    content.extend(attachments)
    return CallToolResult(
        content=content,
        structuredContent=structured.model_dump(mode="json", by_alias=True),
        isError=False,
    )
