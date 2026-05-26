"""Render the eval-suite PR-comment markdown and enforce the pass threshold.

Reads the most recent Inspect AI log under ``--log-dir``, extracts the
hybrid scorer's overall accuracy and per-sample failure reasons, and:

1. Emits a markdown report on stdout (consumed by the ``actions/github-script``
   step that posts a PR comment).
2. Exits with code 0 when accuracy is at or above ``--threshold``, code 1
   otherwise. The workflow uses that exit code to fail the gate.

Kept deliberately small — the CLI shell is thin and the rendering is a
pure function over primitives so the unit tests don't have to construct
synthetic Inspect log objects.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log

_MAX_FAILURES_LISTED = 15


@dataclass(frozen=True)
class FailingPrompt:
    """Per-sample failure record consumed by :func:`render_markdown`."""

    sample_id: str
    trace_passed: bool
    functional_passed: bool
    trace_reasons: tuple[str, ...]
    functional_reasons: tuple[str, ...]

    @property
    def short_reason(self) -> str:
        flags: list[str] = []
        if not self.trace_passed:
            flags.append("trace fail")
        if not self.functional_passed:
            flags.append("functional fail")
        return ", ".join(flags) if flags else "unknown failure mode"


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate of one eval run, ready to render."""

    model: str
    accuracy: float
    n_samples: int
    n_passed: int
    failing: tuple[FailingPrompt, ...]


def collect_failures(log: Any) -> tuple[float, int, int, tuple[FailingPrompt, ...]]:
    """Walk an Inspect AI ``EvalLog`` and pull the bits the report needs.

    Returns ``(accuracy, n_samples, n_passed, failing)``. ``log`` is typed
    ``Any`` because Inspect AI's log types live behind the ``ignore_missing_imports``
    mypy override; we don't need the static type here.
    """
    if log.results is None or not log.results.scores:
        raise ValueError("eval log has no scorer results — run did not complete")
    scorer_result = log.results.scores[0]
    accuracy = float(scorer_result.metrics["accuracy"].value)
    n_samples = int(scorer_result.scored_samples)
    n_passed = round(accuracy * n_samples)

    failing: list[FailingPrompt] = []
    samples = log.samples or []
    for sample in samples:
        # Each sample has at most one score under our scorer name.
        if not sample.scores:
            continue
        score = next(iter(sample.scores.values()))
        if score.value == 1.0:
            continue
        meta = score.metadata or {}
        failing.append(
            FailingPrompt(
                sample_id=str(sample.id),
                trace_passed=bool(meta.get("trace_passed", False)),
                functional_passed=bool(meta.get("functional_passed", False)),
                trace_reasons=tuple(meta.get("trace_failure_reasons") or ()),
                functional_reasons=tuple(meta.get("functional_failure_reasons") or ()),
            )
        )
    return accuracy, n_samples, n_passed, tuple(failing)


def render_markdown(summary: EvalSummary, threshold: float) -> str:
    """Render the PR-comment markdown body. Pure function; easy to unit-test."""
    passed_gate = summary.accuracy >= threshold
    status = "✅ PASS" if passed_gate else "❌ FAIL"

    lines: list[str] = [
        f"## Eval gate: {status}",
        "",
        f"- **Model:** `{summary.model}`",
        (
            f"- **Accuracy:** {summary.accuracy:.3f}  "
            f"({summary.n_passed} / {summary.n_samples} passed)"
        ),
        f"- **Threshold:** ≥ {threshold:.2f}",
        "",
    ]

    if summary.failing:
        lines.append(f"### Failing prompts ({len(summary.failing)})")
        lines.append("")
        for fp in summary.failing[:_MAX_FAILURES_LISTED]:
            lines.append(f"- `{fp.sample_id}` — {fp.short_reason}")
            for reason in fp.trace_reasons[:1]:
                lines.append(f"  - trace: {reason}")
            for reason in fp.functional_reasons[:1]:
                lines.append(f"  - functional: {reason}")
        omitted = len(summary.failing) - _MAX_FAILURES_LISTED
        if omitted > 0:
            lines.append(f"- … and {omitted} more (see workflow artefact for full log)")
        lines.append("")
    else:
        lines.append("Every prompt passed both the trace and functional checks.")
        lines.append("")

    lines.append("Full Inspect log uploaded as the `inspect-eval-logs` workflow artefact.")
    lines.append("")
    return "\n".join(lines)


def _build_summary_from_log_dir(log_dir: Path) -> EvalSummary:
    logs = list_eval_logs(str(log_dir), descending=True)
    if not logs:
        raise FileNotFoundError(f"no Inspect AI logs found under {log_dir}")
    log = read_eval_log(logs[0].name)
    accuracy, n_samples, n_passed, failing = collect_failures(log)
    return EvalSummary(
        model=str(log.eval.model),
        accuracy=accuracy,
        n_samples=n_samples,
        n_passed=n_passed,
        failing=failing,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing Inspect AI .eval logs (default: ./logs).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum accuracy to count as a passing gate (default: 0.80).",
    )
    args = parser.parse_args(argv)

    try:
        summary = _build_summary_from_log_dir(args.log_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        # No log means we can't produce a meaningful report; fail the gate.
        return 2

    sys.stdout.write(render_markdown(summary, args.threshold))
    return 0 if summary.accuracy >= args.threshold else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
