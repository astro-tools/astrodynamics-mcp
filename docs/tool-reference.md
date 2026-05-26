# Tool reference

Every tool the server registers, with the input and output JSON schemas
the MCP wire actually carries. The catalogue below is generated at build
time from the live `FastMCP` registry — what you see is what an MCP
client receives when it issues `tools/list` against `astrodynamics-mcp`.

## How to read this page

Each tool section carries:

- **Description.** The string the LLM sees in its tool catalogue. Tuned
  per the eval suite so the model picks the right tool and binds the
  right arguments under prompt variation.
- **Input schema.** The JSON-Schema the SDK validates every call
  against. The same schema is emitted by `pydantic`'s `.model_json_schema()`
  for the tool's argument model; field `description` and `examples` are
  visible to the LLM via the SDK.
- **Output schema.** The JSON-Schema for the tool's response body. Every
  numeric field follows the `{value, unit}` discipline described under
  [Data sources](data-sources.md#unit-discipline) — there are no bare
  `km` or `km/s` floats on the wire.

For the underlying Python types and the docstrings, see the
[API reference](api.md).

## Tools

<!-- AUTOGEN:tool-reference -->
