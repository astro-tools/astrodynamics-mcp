"""Helper script to regenerate every reference-output golden under ``tests/data/golden/``.

Run from the repo root after an upstream pin bump or a deliberate tool-output
change. Always review the resulting diff under ``tests/data/golden/`` before
committing.

```
uv run python tests/_regenerate_goldens.py
```

The underscore prefix keeps pytest from collecting this module as a test file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from tests._sample_calls import SAMPLE_CALLS, SampleCall

_GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden"


def _mask_volatile_fields(tool_name: str, payload: Any) -> Any:
    """Replace timestamp anchors with the same sentinel the goldens carry."""
    if not isinstance(payload, dict):
        return payload
    if tool_name == "tle_lookup":
        for result in payload.get("results", []):
            if isinstance(result, dict) and "fetched_at" in result:
                result["fetched_at"] = "<masked-in-test>"
    return payload


async def _generate_one(sample: SampleCall) -> dict[str, Any]:
    with sample.setup():
        response = await sample.invoke()
    payload = response.model_dump(mode="json")
    masked: dict[str, Any] = _mask_volatile_fields(sample.tool_name, payload)
    return masked


async def regenerate_all_goldens() -> None:
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLE_CALLS:
        golden_path = _GOLDEN_DIR / f"{sample.tool_name}.json"
        payload = await _generate_one(sample)
        golden_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {golden_path.relative_to(_GOLDEN_DIR.parents[2])}")


if __name__ == "__main__":
    asyncio.run(regenerate_all_goldens())
