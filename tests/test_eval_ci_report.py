"""Tests for ``eval/_ci_report.py``'s markdown renderer and failure collector."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from eval._ci_report import (
    EvalSummary,
    FailingPrompt,
    SkippedPrompt,
    collect_failures,
    main,
    render_markdown,
    render_no_log_markdown,
)


def _fake_log(
    *,
    model: str = "openai-api/github/openai/gpt-4.1-mini",
    samples: list[Any] | None = None,
) -> Any:
    """Build a SimpleNamespace mirroring the slice of EvalLog the helper reads.

    The helper deliberately treats the log as ``Any`` to avoid coupling to
    Inspect AI's internal types — these fakes are enough.
    """
    return SimpleNamespace(
        eval=SimpleNamespace(model=model),
        samples=samples,
    )


def _fake_sample(
    sample_id: str,
    *,
    value: float,
    trace_passed: bool = True,
    functional_passed: bool = True,
    trace_reasons: tuple[str, ...] = (),
    functional_reasons: tuple[str, ...] = (),
) -> Any:
    score = SimpleNamespace(
        value=value,
        metadata={
            "trace_passed": trace_passed,
            "functional_passed": functional_passed,
            "trace_failure_reasons": list(trace_reasons),
            "functional_failure_reasons": list(functional_reasons),
        },
    )
    return SimpleNamespace(id=sample_id, scores={"hybrid_scorer": score}, error=None)


def _fake_errored_sample(sample_id: str, *, message: str) -> Any:
    return SimpleNamespace(
        id=sample_id,
        scores=None,
        error=SimpleNamespace(message=message),
    )


class TestCollectFailures:
    def test_all_passing(self) -> None:
        log = _fake_log(
            samples=[_fake_sample("a", value=1.0), _fake_sample("b", value=1.0)],
        )
        accuracy, n_samples, n_passed, failing = collect_failures(log)
        assert accuracy == 1.0
        assert n_samples == 2
        assert n_passed == 2
        assert failing == ()

    def test_partial_failures(self) -> None:
        log = _fake_log(
            samples=[
                _fake_sample("a", value=1.0),
                _fake_sample(
                    "b",
                    value=0.0,
                    trace_passed=False,
                    functional_passed=False,
                    trace_reasons=("step 0 (tle_lookup): no matching call",),
                    functional_reasons=("$.results: not found",),
                ),
            ],
        )
        accuracy, n_samples, n_passed, failing = collect_failures(log)
        assert accuracy == 0.5
        assert n_samples == 2
        assert n_passed == 1
        assert len(failing) == 1
        f = failing[0]
        assert f.sample_id == "b"
        assert f.trace_passed is False
        assert f.functional_passed is False
        assert f.error is None

    def test_passes_score_not_one(self) -> None:
        # Score==1.0 means pass in our hybrid scorer; anything else fails.
        log = _fake_log(
            samples=[
                _fake_sample(
                    "a",
                    value=0.0,
                    trace_passed=True,
                    functional_passed=False,
                    functional_reasons=("$.results[0].norad_id: expected '25544', got '00000'",),
                )
            ],
        )
        _, _, _, failing = collect_failures(log)
        assert len(failing) == 1
        assert failing[0].trace_passed is True
        assert failing[0].functional_passed is False

    def test_errored_sample_counted_as_failure(self) -> None:
        # An errored sample (e.g. 413 tokens_limit_reached) has no score
        # but counts against accuracy as if it scored 0, and its error
        # message surfaces as the failure reason.
        log = _fake_log(
            samples=[
                _fake_sample("scored_pass", value=1.0),
                _fake_errored_sample(
                    "porkchop_blew_token_cap",
                    message="Error code: 413 - tokens_limit_reached",
                ),
            ],
        )
        accuracy, n_samples, n_passed, failing = collect_failures(log)
        assert n_samples == 2
        assert n_passed == 1
        assert accuracy == 0.5
        assert len(failing) == 1
        assert failing[0].sample_id == "porkchop_blew_token_cap"
        assert failing[0].error is not None
        assert "tokens_limit_reached" in failing[0].error
        assert failing[0].short_reason == "sample errored"

    def test_empty_log_raises(self) -> None:
        log = SimpleNamespace(samples=None)
        with pytest.raises(ValueError, match="no samples"):
            collect_failures(log)


class TestRenderMarkdown:
    def _summary(self, **overrides: Any) -> EvalSummary:
        defaults: dict[str, Any] = {
            "model": "openai-api/github/openai/gpt-4o",
            "accuracy": 1.0,
            "n_samples": 30,
            "n_passed": 30,
            "failing": (),
        }
        defaults.update(overrides)
        return EvalSummary(**defaults)

    def test_pass_path(self) -> None:
        md = render_markdown(self._summary(), threshold=0.80)
        assert "## Eval gate: ✅ PASS" in md
        assert "**Accuracy:** 1.000" in md
        assert "Every prompt that ran passed" in md

    def test_fail_path_lists_failures(self) -> None:
        failing = (
            FailingPrompt(
                sample_id="tle_lookup_iss_by_norad_id",
                trace_passed=False,
                functional_passed=True,
                trace_reasons=("step 0 (tle_lookup): no matching call",),
                functional_reasons=(),
            ),
            FailingPrompt(
                sample_id="sgp4_propagate_iss_default_teme",
                trace_passed=True,
                functional_passed=False,
                trace_reasons=(),
                functional_reasons=("$.states[0].r.value: expected l2_in_range=[6500, 7500]",),
            ),
        )
        summary = self._summary(accuracy=28 / 30, n_passed=28, failing=failing)
        md = render_markdown(summary, threshold=0.80)
        assert "## Eval gate: ✅ PASS" in md  # 28/30 = 0.93 still above 0.80
        assert "tle_lookup_iss_by_norad_id" in md
        assert "trace fail" in md
        assert "functional fail" in md
        assert "step 0 (tle_lookup): no matching call" in md
        assert "expected l2_in_range" in md

    def test_below_threshold_marks_fail(self) -> None:
        failing = tuple(
            FailingPrompt(
                sample_id=f"prompt_{i}",
                trace_passed=False,
                functional_passed=False,
                trace_reasons=("trace bust",),
                functional_reasons=(),
            )
            for i in range(10)
        )
        summary = self._summary(accuracy=20 / 30, n_passed=20, failing=failing)
        md = render_markdown(summary, threshold=0.80)
        assert "## Eval gate: ❌ FAIL" in md

    def test_errored_section_renders(self) -> None:
        failing = (
            FailingPrompt(
                sample_id="porkchop_blew_token_cap",
                trace_passed=False,
                functional_passed=False,
                trace_reasons=(),
                functional_reasons=(),
                error="Error code: 413 - tokens_limit_reached",
            ),
        )
        summary = self._summary(accuracy=20 / 21, n_samples=21, n_passed=20, failing=failing)
        md = render_markdown(summary, threshold=0.80)
        assert "**Errored samples:** 1 (counted as failures)" in md
        assert "sample errored" in md
        assert "error: Error code: 413" in md

    def test_skipped_section_renders(self) -> None:
        skipped = (
            SkippedPrompt(
                sample_id="sequential_spacetrack_tle_then_sgp4", unmet=("credential:spacetrack",)
            ),
            SkippedPrompt(sample_id="gmat_run_mission_simple_orbit", unmet=("gmat",)),
        )
        summary = self._summary(skipped=skipped)
        md = render_markdown(summary, threshold=0.80)
        assert "**Skipped:** 2" in md
        assert "### Skipped prompts (2)" in md
        assert "sequential_spacetrack_tle_then_sgp4" in md
        assert "unmet: credential:spacetrack" in md
        assert "gmat_run_mission_simple_orbit" in md

    def test_no_skipped_section_when_empty(self) -> None:
        md = render_markdown(self._summary(), threshold=0.80)
        assert "Skipped" not in md

    def test_skips_do_not_affect_gate(self) -> None:
        # Accuracy (over run prompts) drives the gate; skips are neutral.
        skipped = (SkippedPrompt(sample_id="x", unmet=("gmat",)),)
        summary = self._summary(accuracy=0.9, n_samples=10, n_passed=9, skipped=skipped)
        md = render_markdown(summary, threshold=0.80)
        assert "## Eval gate: ✅ PASS" in md

    def test_truncates_long_failure_list(self) -> None:
        failing = tuple(
            FailingPrompt(
                sample_id=f"prompt_{i}",
                trace_passed=False,
                functional_passed=False,
                trace_reasons=("trace bust",),
                functional_reasons=(),
            )
            for i in range(25)
        )
        summary = self._summary(accuracy=5 / 30, n_passed=5, failing=failing)
        md = render_markdown(summary, threshold=0.80)
        # Truncation marker present, and not every prompt is in the body.
        assert "and 10 more" in md
        assert "`prompt_0`" in md
        assert "`prompt_24`" not in md


class TestNoLogMarkdown:
    def test_includes_reason(self) -> None:
        md = render_no_log_markdown("no Inspect AI logs found under logs")
        assert "## Eval gate: ❌ ERROR" in md
        assert "no Inspect AI logs found" in md


class TestMainCli:
    def test_missing_log_dir_returns_two_and_writes_body(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["--log-dir", str(tmp_path), "--threshold", "0.8"])
        assert rc == 2
        captured = capsys.readouterr()
        # Body lands on stdout (run-summary source); reason on stderr (workflow log).
        assert "ERROR" in captured.out
        assert "ERROR" in captured.err

    def test_threshold_pass_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub _build_summary_from_log_dir so we don't need a real .eval file.
        from eval import _ci_report

        def stub(_log_dir: Path) -> EvalSummary:
            return EvalSummary(model="x", accuracy=0.9, n_samples=30, n_passed=27, failing=())

        monkeypatch.setattr(_ci_report, "_build_summary_from_log_dir", stub)
        rc = main(["--log-dir", str(tmp_path), "--threshold", "0.80"])
        assert rc == 0

    def test_threshold_fail_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from eval import _ci_report

        def stub(_log_dir: Path) -> EvalSummary:
            return EvalSummary(model="x", accuracy=0.5, n_samples=30, n_passed=15, failing=())

        monkeypatch.setattr(_ci_report, "_build_summary_from_log_dir", stub)
        rc = main(["--log-dir", str(tmp_path), "--threshold", "0.80"])
        assert rc == 1
