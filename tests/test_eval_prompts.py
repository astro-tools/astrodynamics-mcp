"""Tests for the eval-suite prompt YAML loader and schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval._prompts import PromptSpec, load_prompt_from_yaml, load_prompts

_VALID_YAML = """
prompt: "Fetch the current TLE for the ISS (NORAD 25544)."
tier: single_tool
tools_required: [tle_lookup]
permitted_traces:
  - - tool: tle_lookup
      arg_constraints:
        query: {equals: "25544"}
functional_answer:
  - path: "$.results[0].norad_id"
    equals: "25544"
  - path: "$.results"
    length: {min: 1}
notes: "Primary single-tool prompt for tle_lookup."
"""


def _write(tmp: Path, name: str, content: str) -> Path:
    path = tmp / name
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadPromptFromYaml:
    def test_valid_prompt_parses(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "iss_tle_lookup.yaml", _VALID_YAML)
        prompt = load_prompt_from_yaml(path)
        assert isinstance(prompt, PromptSpec)
        assert prompt.id == "iss_tle_lookup"
        assert prompt.tier == "single_tool"
        assert prompt.tools_required == ["tle_lookup"]
        assert prompt.permitted_traces[0][0].tool == "tle_lookup"
        assert prompt.permitted_traces[0][0].arg_constraints == {"query": {"equals": "25544"}}

    def test_id_defaults_to_stem(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "weather_group.yaml", _VALID_YAML)
        prompt = load_prompt_from_yaml(path)
        assert prompt.id == "weather_group"

    def test_explicit_id_wins(self, tmp_path: Path) -> None:
        explicit = "id: my-custom-id\n" + _VALID_YAML
        path = _write(tmp_path, "anything.yaml", explicit)
        prompt = load_prompt_from_yaml(path)
        assert prompt.id == "my-custom-id"

    def test_unknown_top_level_field_rejected(self, tmp_path: Path) -> None:
        bad = _VALID_YAML + "\nunknown_field: 42\n"
        path = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(Exception, match=r"(?i)unknown_field|extra"):
            load_prompt_from_yaml(path)

    def test_empty_tools_required_rejected(self, tmp_path: Path) -> None:
        bad = _VALID_YAML.replace("tools_required: [tle_lookup]", "tools_required: []")
        path = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(Exception, match="at least one tool"):
            load_prompt_from_yaml(path)

    def test_empty_permitted_traces_rejected(self, tmp_path: Path) -> None:
        old_traces = (
            "permitted_traces:\n"
            "  - - tool: tle_lookup\n"
            "      arg_constraints:\n"
            '        query: {equals: "25544"}'
        )
        bad = _VALID_YAML.replace(old_traces, "permitted_traces: []")
        path = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(Exception, match="at least one trace"):
            load_prompt_from_yaml(path)

    def test_tool_outside_tools_required_rejected(self, tmp_path: Path) -> None:
        bad = _VALID_YAML.replace("tool: tle_lookup", "tool: sgp4_propagate")
        path = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(Exception, match="not in tools_required"):
            load_prompt_from_yaml(path)

    def test_invalid_arg_constraint_rejected(self, tmp_path: Path) -> None:
        bad = _VALID_YAML.replace(
            'query: {equals: "25544"}',
            'query: {unknown_predicate: "25544"}',
        )
        path = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(Exception, match="unknown constraint predicate"):
            load_prompt_from_yaml(path)

    def test_invalid_functional_check_rejected(self, tmp_path: Path) -> None:
        bad = _VALID_YAML.replace(
            '- path: "$.results[0].norad_id"\n    equals: "25544"',
            '- path: "$.results[0].norad_id"\n    weird_predicate: 1',
        )
        path = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(Exception, match="unknown functional predicate"):
            load_prompt_from_yaml(path)

    def test_empty_yaml_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "empty.yaml", "")
        with pytest.raises(Exception, match=r"(?i)empty"):
            load_prompt_from_yaml(path)


class TestLoadPrompts:
    def test_skips_underscore_prefix_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "real_prompt.yaml", _VALID_YAML)
        _write(tmp_path, "_fixture.yaml", _VALID_YAML)
        prompts = load_prompts(tmp_path)
        assert len(prompts) == 1
        assert prompts[0].id == "real_prompt"

    def test_returns_empty_when_directory_missing(self, tmp_path: Path) -> None:
        prompts = load_prompts(tmp_path / "does-not-exist")
        assert prompts == []

    def test_sorted_by_filename(self, tmp_path: Path) -> None:
        _write(tmp_path, "b.yaml", _VALID_YAML)
        _write(tmp_path, "a.yaml", _VALID_YAML)
        prompts = load_prompts(tmp_path)
        assert [p.id for p in prompts] == ["a", "b"]
