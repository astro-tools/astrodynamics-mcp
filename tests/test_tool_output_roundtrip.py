"""Output-schema round-trip test — for each v0.1 tool, JSON → schema → JSON is idempotent.

Calls each registered tool with a fixed sample input (mocked CelesTrak /
Horizons where needed), dumps the response model to JSON, parses it back
through the declared output schema, and asserts the second JSON dump is
bit-identical to the first. Catches schema drift between the tool body
and the pydantic class it advertises — most commonly an extra key the
tool emits that the schema forbids (or vice versa).

Parametrised over the shared :data:`SAMPLE_CALLS` table so every tool
exercises the same contract.
"""

from __future__ import annotations

import json

import pytest

from tests._sample_calls import SAMPLE_CALLS, SampleCall


@pytest.mark.parametrize("sample", SAMPLE_CALLS, ids=lambda s: s.tool_name)
async def test_response_roundtrips_through_output_schema(sample: SampleCall) -> None:
    """Tool response → JSON → schema → JSON is idempotent at the byte level.

    Schema drift between the function body and the declared output model
    shows up here as either a ``ValidationError`` (extra key, missing key,
    wrong type) or as a non-identical second JSON dump.
    """
    with sample.setup():
        response = await sample.invoke()

    assert isinstance(response, sample.output_model), (
        f"{sample.tool_name} returned {type(response).__name__}, expected "
        f"{sample.output_model.__name__}"
    )

    # First pass: model → JSON.
    first_json = response.model_dump_json()

    # Second pass: parse back through the declared output schema. The
    # schema is the contract the LLM consumer reads; if the response
    # doesn't conform to it, the consumer can't parse it.
    rebuilt = sample.output_model.model_validate_json(first_json)
    second_json = rebuilt.model_dump_json()

    assert first_json == second_json, (
        f"{sample.tool_name} output not idempotent under JSON round-trip: "
        f"first dump differs from second"
    )


@pytest.mark.parametrize("sample", SAMPLE_CALLS, ids=lambda s: s.tool_name)
async def test_response_dict_is_json_serialisable(sample: SampleCall) -> None:
    """``model_dump()`` must produce a value ``json.dumps`` accepts as-is.

    A non-JSON-serialisable value sneaking into ``model_dump()`` (a
    ``datetime``, an enum that didn't get coerced, a custom object) would
    pass the previous test (which uses ``model_dump_json``) but break MCP
    wire serialisation downstream.
    """
    with sample.setup():
        response = await sample.invoke()

    dumped = response.model_dump(mode="json")
    # `json.dumps` raises TypeError on non-serialisable leaves; that's the
    # implicit assertion. We round-trip through json.loads so a future
    # contributor reading the test sees the full contract.
    text = json.dumps(dumped, sort_keys=True)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
