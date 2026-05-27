"""Tests for `astrodynamics_mcp.runs`.

Covers the in-memory LRU semantics, the disk-index round-trip,
cross-process replay on a fresh registry instance, and the disabled-
mode behaviour shared with :class:`~astrodynamics_mcp.cache.Cache`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrodynamics_mcp.runs import (
    RunArtefacts,
    RunRegistry,
    default_registry,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Per-test parent directory for synthetic ``output_dir``s."""
    workspaces = tmp_path / "workspaces"
    workspaces.mkdir()
    return workspaces


def _make_workspace(parent: Path, slug: str) -> Path:
    """Materialise a synthetic output dir with one artefact file."""
    path = parent / slug
    path.mkdir()
    (path / "ReportFile1.txt").write_text("header\nrow\n", encoding="utf-8")
    return path


class TestMint:
    def test_returns_hex_uuid4(self) -> None:
        reg = RunRegistry(directory=None, limit=5)
        run_id = reg.mint()
        # UUID4 hex is 32 lowercase hex chars; verify the shape rather
        # than the value (which is random).
        assert len(run_id) == 32
        assert all(c in "0123456789abcdef" for c in run_id)

    def test_mints_are_unique(self) -> None:
        reg = RunRegistry(directory=None, limit=5)
        ids = {reg.mint() for _ in range(100)}
        assert len(ids) == 100


class TestRegisterAndGet:
    def test_register_then_get_returns_entry(self, tmp_path: Path, workspace: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=5)
        out = _make_workspace(workspace, "run-a")
        run_id = reg.mint()
        reg.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        got = reg.get(run_id)
        assert got is not None
        assert got.run_id == run_id
        assert got.output_dir == out
        assert got.artefacts == {"ReportFile1": out / "ReportFile1.txt"}

    def test_get_unknown_returns_none(self, tmp_path: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=5)
        assert reg.get("does-not-exist") is None

    def test_register_writes_disk_index(self, tmp_path: Path, workspace: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=5)
        out = _make_workspace(workspace, "run-a")
        run_id = reg.mint()
        reg.register(
            run_id,
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
        )
        index_path = tmp_path / "runs" / f"{run_id}.json"
        assert index_path.is_file()
        payload = json.loads(index_path.read_text())
        assert payload["run_id"] == run_id
        assert payload["output_dir"] == str(out)
        assert payload["artefacts"] == {"ReportFile1": str(out / "ReportFile1.txt")}
        assert "created_at" in payload

    def test_known_run_ids_in_creation_order(self, tmp_path: Path, workspace: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=5)
        ids = []
        for i in range(3):
            out = _make_workspace(workspace, f"run-{i}")
            rid = reg.mint()
            reg.register(rid, output_dir=out, artefacts={})
            ids.append(rid)
        assert reg.known_run_ids() == ids


class TestEviction:
    def test_oldest_evicted_when_over_limit(self, tmp_path: Path, workspace: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=2)
        out_a = _make_workspace(workspace, "run-a")
        out_b = _make_workspace(workspace, "run-b")
        out_c = _make_workspace(workspace, "run-c")
        id_a, id_b, id_c = reg.mint(), reg.mint(), reg.mint()
        reg.register(id_a, output_dir=out_a, artefacts={})
        reg.register(id_b, output_dir=out_b, artefacts={})
        # Third entry trips the cap; the oldest (a) should be gone.
        reg.register(id_c, output_dir=out_c, artefacts={})
        assert reg.get(id_a) is None
        assert reg.get(id_b) is not None
        assert reg.get(id_c) is not None
        # The on-disk output dir for the evicted entry is gone too.
        assert not out_a.exists()
        # As is its index file.
        assert not (tmp_path / "runs" / f"{id_a}.json").exists()

    def test_eviction_survives_missing_output_dir(self, tmp_path: Path, workspace: Path) -> None:
        """A peer process / OS reaper may have removed ``output_dir`` first.

        Eviction must not raise in that case — ``shutil.rmtree`` with
        ``ignore_errors=True`` is what the registry promises.
        """
        reg = RunRegistry(directory=tmp_path, limit=1)
        out_a = _make_workspace(workspace, "run-a")
        id_a = reg.mint()
        reg.register(id_a, output_dir=out_a, artefacts={})
        # Wipe the dir behind the registry's back, then trigger eviction.
        import shutil

        shutil.rmtree(out_a)
        out_b = _make_workspace(workspace, "run-b")
        reg.register(reg.mint(), output_dir=out_b, artefacts={})
        # Survived, and the prior entry is no longer reachable.
        assert reg.get(id_a) is None

    def test_register_with_invalid_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            RunRegistry(directory=None, limit=0)
        with pytest.raises(ValueError, match="must be >= 1"):
            RunRegistry(directory=None, limit=-3)

    def test_duplicate_run_id_replaces_payload(self, tmp_path: Path, workspace: Path) -> None:
        """Re-registering the same id drops the prior dir + index.

        UUID4 collisions are vanishingly unlikely, but the contract must
        hold: the prior payload is reclaimed, never silently leaked.
        """
        reg = RunRegistry(directory=tmp_path, limit=5)
        out_a = _make_workspace(workspace, "run-a")
        out_b = _make_workspace(workspace, "run-b")
        rid = reg.mint()
        reg.register(rid, output_dir=out_a, artefacts={})
        reg.register(rid, output_dir=out_b, artefacts={})
        assert not out_a.exists()
        assert reg.get(rid) is not None
        assert reg.get(rid).output_dir == out_b  # type: ignore[union-attr]


class TestDrop:
    def test_drop_removes_entry_and_payload(self, tmp_path: Path, workspace: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=5)
        out = _make_workspace(workspace, "run-a")
        rid = reg.mint()
        reg.register(rid, output_dir=out, artefacts={"ReportFile1": out / "ReportFile1.txt"})

        assert reg.drop(rid) is True
        assert reg.get(rid) is None
        assert reg.known_run_ids() == []
        # Index JSON is gone …
        assert not (tmp_path / "runs" / f"{rid}.json").exists()
        # … and the output dir is rmtreed.
        assert not out.exists()

    def test_drop_unknown_id_is_noop(self, tmp_path: Path) -> None:
        reg = RunRegistry(directory=tmp_path, limit=5)
        assert reg.drop("does-not-exist") is False

    def test_drop_survives_already_missing_output_dir(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        """The motivating case: an external reaper got there first."""
        reg = RunRegistry(directory=tmp_path, limit=5)
        out = _make_workspace(workspace, "run-a")
        rid = reg.mint()
        reg.register(rid, output_dir=out, artefacts={"ReportFile1": out / "ReportFile1.txt"})
        import shutil

        shutil.rmtree(out)
        # Should not raise.
        assert reg.drop(rid) is True
        assert reg.get(rid) is None


class TestLimitResolution:
    def test_constructor_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT", "10")
        reg = RunRegistry(directory=None, limit=3)
        assert reg.limit == 3

    def test_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT", "7")
        reg = RunRegistry(directory=None)
        assert reg.limit == 7

    def test_default_is_50(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT", raising=False)
        reg = RunRegistry(directory=None)
        assert reg.limit == 50

    def test_bad_env_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT", "not-a-number")
        with pytest.raises(ValueError, match="is not an integer"):
            RunRegistry(directory=None)

    def test_negative_env_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_RUN_REGISTRY_LIMIT", "-1")
        with pytest.raises(ValueError, match="must be >= 1"):
            RunRegistry(directory=None)


class TestCrossRestartReplay:
    def test_fresh_registry_recovers_persisted_entries(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        first = RunRegistry(directory=tmp_path, limit=5)
        out_a = _make_workspace(workspace, "run-a")
        out_b = _make_workspace(workspace, "run-b")
        id_a, id_b = first.mint(), first.mint()
        first.register(id_a, output_dir=out_a, artefacts={"ReportFile1": out_a / "ReportFile1.txt"})
        first.register(id_b, output_dir=out_b, artefacts={})
        # Simulate process restart: build a fresh registry against the
        # same cache root.
        second = RunRegistry(directory=tmp_path, limit=5)
        assert {rid for rid in second.known_run_ids()} == {id_a, id_b}
        got_a = second.get(id_a)
        assert got_a is not None
        assert got_a.artefacts == {"ReportFile1": out_a / "ReportFile1.txt"}

    def test_missing_output_dir_drops_entry_on_replay(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        first = RunRegistry(directory=tmp_path, limit=5)
        out_a = _make_workspace(workspace, "run-a")
        out_b = _make_workspace(workspace, "run-b")
        id_a, id_b = first.mint(), first.mint()
        first.register(id_a, output_dir=out_a, artefacts={})
        first.register(id_b, output_dir=out_b, artefacts={})
        # Simulate OS-level reaping of the bytes for run-a.
        import shutil

        shutil.rmtree(out_a)
        second = RunRegistry(directory=tmp_path, limit=5)
        assert second.known_run_ids() == [id_b]
        # The orphan JSON was cleaned up during replay.
        assert not (tmp_path / "runs" / f"{id_a}.json").exists()

    def test_replay_evicts_when_limit_shrank(self, tmp_path: Path, workspace: Path) -> None:
        first = RunRegistry(directory=tmp_path, limit=5)
        ids = []
        for i in range(4):
            out = _make_workspace(workspace, f"run-{i}")
            rid = first.mint()
            first.register(rid, output_dir=out, artefacts={})
            ids.append(rid)
        # Operator restarts with a tighter cap. The two oldest entries
        # must be evicted on first access in the new process.
        second = RunRegistry(directory=tmp_path, limit=2)
        recovered = second.known_run_ids()
        assert recovered == ids[-2:]
        # Their on-disk dirs are gone too.
        for rid in ids[:2]:
            assert not (workspace / f"run-{ids.index(rid)}").exists()

    def test_corrupt_index_file_ignored(self, tmp_path: Path, workspace: Path) -> None:
        first = RunRegistry(directory=tmp_path, limit=5)
        out_a = _make_workspace(workspace, "run-a")
        id_a = first.mint()
        first.register(id_a, output_dir=out_a, artefacts={})
        # Drop a corrupt sibling JSON.
        (tmp_path / "runs" / "garbage.json").write_text("{not json")
        second = RunRegistry(directory=tmp_path, limit=5)
        assert second.known_run_ids() == [id_a]


class TestDisabledMode:
    def test_empty_env_disables_disk_index(
        self, tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        reg = RunRegistry()
        assert reg.directory is None
        out = _make_workspace(workspace, "run-a")
        rid = reg.mint()
        reg.register(rid, output_dir=out, artefacts={})
        # Memory works …
        assert reg.get(rid) is not None
        # … but nothing was written to disk.
        # (tmp_path is the test fixture; the runs/ subdir would land
        # under the resolved cache root, which is None in disabled mode.
        # We can still check there's no orphan dir created.)
        assert not (tmp_path / "runs").exists()

    def test_disabled_mode_does_not_persist_across_instances(
        self, monkeypatch: pytest.MonkeyPatch, workspace: Path
    ) -> None:
        monkeypatch.setenv("ASTRODYNAMICS_MCP_CACHE_DIR", "")
        reg1 = RunRegistry()
        out = _make_workspace(workspace, "run-a")
        rid = reg1.mint()
        reg1.register(rid, output_dir=out, artefacts={})
        reg2 = RunRegistry()
        assert reg2.get(rid) is None


class TestRunArtefactsValue:
    def test_frozen(self, workspace: Path) -> None:
        import dataclasses
        from datetime import datetime, timezone

        out = _make_workspace(workspace, "run-a")
        entry = RunArtefacts(
            run_id="abc",
            output_dir=out,
            artefacts={"ReportFile1": out / "ReportFile1.txt"},
            created_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.run_id = "different"  # type: ignore[misc]


class TestDefaultRegistrySingleton:
    def test_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import astrodynamics_mcp.runs as runs_module

        monkeypatch.setattr(runs_module, "_default_registry", None)
        r1 = default_registry()
        r2 = default_registry()
        assert r1 is r2
