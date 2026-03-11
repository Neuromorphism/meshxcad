"""Tests for federated learning shard/compact system."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from meshxcad.optim import (
    SegmentationStrategySelector,
    FixerPrioritySelector,
    STRATEGY_NAMES,
    FEATURE_NAMES,
    mesh_features,
    _features_to_vector,
)
from meshxcad.federation import (
    record_experience,
    record_segmentation_experience,
    record_fixer_experience,
    get_session_experiences,
    clear_session_experiences,
    save_shard,
    load_shard,
    list_shards,
    save_canonical,
    load_canonical,
    compact,
    _session_experiences,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_sphere_features():
    """Return a realistic feature vector for a sphere-like mesh."""
    return [0.9, 0.95, 0.5, 0.1, 0.1, 0.3]


def _make_elongated_features():
    """Return a realistic feature vector for a tall/elongated mesh."""
    return [3.0, 0.3, 0.6, 0.2, 0.8, 0.1]


FIXER_NAMES = ["hausdorff", "silhouette", "curvature", "volume", "edge_length"]


# ── Experience Recording ────────────────────────────────────────────────

class TestExperienceRecording:
    def setup_method(self):
        clear_session_experiences()

    def test_record_segmentation(self):
        record_segmentation_experience(
            _make_sphere_features(), "sdf", 0.95)
        exps = get_session_experiences()
        assert len(exps) == 1
        assert exps[0]["kind"] == "segmentation"
        assert exps[0]["strategy"] == "sdf"
        assert exps[0]["quality"] == 0.95
        assert len(exps[0]["features"]) == 6

    def test_record_fixer(self):
        record_fixer_experience(
            FIXER_NAMES, "hausdorff",
            {"hausdorff": 10.0, "silhouette": 5.0},
            {"silhouette"}, {"hausdorff": {"avg_improvement": 60.0}},
            improved=True, improvement_amount=8.0)
        exps = get_session_experiences()
        assert len(exps) == 1
        assert exps[0]["kind"] == "fixer"
        assert exps[0]["chosen"] == "hausdorff"
        assert exps[0]["improved"] is True

    def test_multiple_records(self):
        for i in range(5):
            record_segmentation_experience(
                _make_sphere_features(), "sdf", 0.9 + i * 0.01)
        assert len(get_session_experiences()) == 5

    def test_clear(self):
        record_segmentation_experience(
            _make_sphere_features(), "sdf", 0.9)
        clear_session_experiences()
        assert len(get_session_experiences()) == 0

    def test_auto_recording_from_selector(self):
        """SegmentationStrategySelector.update() auto-records."""
        clear_session_experiences()
        sel = SegmentationStrategySelector()
        # Make a simple mesh
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                      dtype=float)
        f = np.array([[0, 1, 2], [1, 2, 4], [0, 1, 3], [1, 3, 5],
                       [2, 4, 6], [4, 6, 7], [3, 5, 6], [5, 6, 7]],
                      dtype=int)
        sel.update(v, f, "sdf", 0.8)
        exps = get_session_experiences()
        seg_exps = [e for e in exps if e["kind"] == "segmentation"]
        assert len(seg_exps) >= 1
        assert seg_exps[-1]["strategy"] == "sdf"

    def test_auto_recording_from_fixer_selector(self):
        """FixerPrioritySelector.update() auto-records."""
        clear_session_experiences()
        sel = FixerPrioritySelector(FIXER_NAMES)
        sel.update(
            "hausdorff",
            {"hausdorff": 10.0, "silhouette": 5.0},
            set(), {"hausdorff": {"avg_improvement": 50.0}},
            improved=True, improvement_amount=7.0)
        exps = get_session_experiences()
        fixer_exps = [e for e in exps if e["kind"] == "fixer"]
        assert len(fixer_exps) >= 1


# ── Shard I/O ───────────────────────────────────────────────────────────

class TestShardIO:
    def setup_method(self):
        clear_session_experiences()

    def test_save_and_load(self, tmp_path):
        for i in range(3):
            record_segmentation_experience(
                _make_sphere_features(), "sdf", 0.9 + i * 0.01)
        record_fixer_experience(
            FIXER_NAMES, "hausdorff",
            {"hausdorff": 10.0}, set(),
            {"hausdorff": {"avg_improvement": 50.0}},
            True, 5.0)

        path = save_shard(tmp_path / "test.jsonl")
        assert path is not None
        assert path.exists()

        loaded = load_shard(path)
        assert len(loaded) == 4
        assert loaded[0]["kind"] == "segmentation"
        assert loaded[3]["kind"] == "fixer"

    def test_save_empty_returns_none(self, tmp_path):
        path = save_shard(tmp_path / "empty.jsonl")
        assert path is None

    def test_jsonl_format(self, tmp_path):
        record_segmentation_experience([1.0] * 6, "skeleton", 0.5)
        path = save_shard(tmp_path / "fmt.jsonl")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["kind"] == "segmentation"

    def test_list_shards(self, tmp_path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        for name in ["a.jsonl", "b.jsonl", "c.txt", "d.jsonl"]:
            (shard_dir / name).touch()
        shards = list_shards(shard_dir)
        assert len(shards) == 3
        assert all(s.suffix == ".jsonl" for s in shards)

    def test_list_shards_empty(self, tmp_path):
        assert list_shards(tmp_path / "nonexistent") == []


# ── Canonical Weight I/O ────────────────────────────────────────────────

class TestCanonicalIO:
    def test_save_and_load_segmentation(self, tmp_path):
        sel = SegmentationStrategySelector()
        W, b = sel.get_weights()
        W[0, 0] = 99.0  # distinctive value
        sel.set_weights(W, b)

        canon_path = tmp_path / "canonical.npz"
        save_canonical(sel, path=canon_path)
        assert canon_path.exists()

        # Load into a fresh selector
        sel2 = SegmentationStrategySelector()
        loaded = load_canonical(seg_selector=sel2, path=canon_path)
        assert loaded is True

        W2, b2 = sel2.get_weights()
        np.testing.assert_allclose(W2[0, 0], 99.0)

    def test_save_and_load_fixer(self, tmp_path):
        fixer_sel = FixerPrioritySelector(FIXER_NAMES)
        seg_sel = SegmentationStrategySelector()

        canon_path = tmp_path / "canonical.npz"
        save_canonical(seg_sel, fixer_sel, path=canon_path)

        fixer_sel2 = FixerPrioritySelector(FIXER_NAMES)
        load_canonical(
            seg_selector=seg_sel, fixer_selector=fixer_sel2,
            path=canon_path)
        # Should not raise

    def test_load_nonexistent(self, tmp_path):
        sel = SegmentationStrategySelector()
        loaded = load_canonical(
            seg_selector=sel, path=tmp_path / "nope.npz")
        assert loaded is False


# ── Compaction ──────────────────────────────────────────────────────────

class TestCompaction:
    def _write_seg_shard(self, shard_dir, n=10, strategy="sdf",
                          quality=0.95):
        """Write a shard with segmentation experiences."""
        clear_session_experiences()
        for _ in range(n):
            record_segmentation_experience(
                _make_sphere_features(), strategy, quality)
        return save_shard(shard_dir / f"seg_{strategy}_{time.time()}.jsonl")

    def _write_fixer_shard(self, shard_dir, n=5, chosen="hausdorff",
                            improved=True, amount=5.0):
        clear_session_experiences()
        for _ in range(n):
            record_fixer_experience(
                FIXER_NAMES, chosen,
                {name: 10.0 for name in FIXER_NAMES},
                set(), {name: {"avg_improvement": 50.0}
                        for name in FIXER_NAMES},
                improved, amount)
        return save_shard(shard_dir / f"fix_{chosen}_{time.time()}.jsonl")

    def test_compact_no_shards(self, tmp_path):
        stats = compact(
            shards_dir=tmp_path / "empty",
            output_path=tmp_path / "out.npz")
        assert stats["status"] == "no_shards"

    def test_compact_segmentation_only(self, tmp_path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._write_seg_shard(shard_dir, n=20, strategy="sdf", quality=1.0)

        canon = tmp_path / "canonical.npz"
        stats = compact(
            shards_dir=shard_dir, output_path=canon,
            epochs=2, shuffle=False)

        assert stats["status"] == "ok"
        assert stats["segmentation_experiences"] == 20
        assert stats["shards_processed"] == 1
        assert canon.exists()

        # Verify the weights were updated
        sel = SegmentationStrategySelector()
        load_canonical(seg_selector=sel, path=canon)
        W, b = sel.get_weights()
        # After many "sdf is great" experiences, sdf row should be strong
        sdf_idx = STRATEGY_NAMES.index("sdf")
        assert W[sdf_idx].sum() != 0  # weights moved from init

    def test_compact_fixer_only(self, tmp_path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._write_fixer_shard(shard_dir, n=10, chosen="hausdorff",
                                 improved=True, amount=10.0)

        canon = tmp_path / "canonical.npz"
        stats = compact(
            shards_dir=shard_dir, output_path=canon, epochs=1)

        assert stats["status"] == "ok"
        assert stats["fixer_experiences"] == 10

    def test_compact_mixed(self, tmp_path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._write_seg_shard(shard_dir, n=5)
        self._write_fixer_shard(shard_dir, n=5)

        canon = tmp_path / "canonical.npz"
        stats = compact(
            shards_dir=shard_dir, output_path=canon, epochs=1)

        assert stats["status"] == "ok"
        assert stats["total_experiences"] == 10

    def test_compact_multiple_shards(self, tmp_path):
        """Multiple shard files from different 'sessions'."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        # Simulate 3 remote sessions
        self._write_seg_shard(shard_dir, n=10, strategy="sdf", quality=0.9)
        self._write_seg_shard(shard_dir, n=10, strategy="skeleton",
                               quality=0.8)
        self._write_seg_shard(shard_dir, n=10, strategy="convexity",
                               quality=0.7)

        canon = tmp_path / "canonical.npz"
        stats = compact(
            shards_dir=shard_dir, output_path=canon, epochs=3)

        assert stats["shards_processed"] == 3
        assert stats["segmentation_experiences"] == 30

    def test_compact_archive(self, tmp_path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._write_seg_shard(shard_dir, n=3)

        canon = tmp_path / "canonical.npz"
        stats = compact(
            shards_dir=shard_dir, output_path=canon,
            epochs=1, archive=True)

        assert stats.get("archived") == 1
        # Original shard gone, moved to archive/
        assert len(list_shards(shard_dir)) == 0
        archive_dir = shard_dir / "archive"
        assert len(list(archive_dir.glob("*.jsonl"))) == 1

    def test_incremental_compaction(self, tmp_path):
        """Compacting twice should build on previous canonical."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        canon = tmp_path / "canonical.npz"

        # First round
        self._write_seg_shard(shard_dir, n=20, strategy="sdf", quality=1.0)
        compact(shards_dir=shard_dir, output_path=canon,
                epochs=2, archive=True)

        sel1 = SegmentationStrategySelector()
        load_canonical(seg_selector=sel1, path=canon)
        W1, _ = sel1.get_weights()

        # Second round with different data
        self._write_seg_shard(shard_dir, n=20, strategy="skeleton",
                               quality=1.0)
        compact(shards_dir=shard_dir, output_path=canon,
                epochs=2, archive=True)

        sel2 = SegmentationStrategySelector()
        load_canonical(seg_selector=sel2, path=canon)
        W2, _ = sel2.get_weights()

        # Weights should have shifted from round 1
        assert not np.allclose(W1, W2)

    def test_deterministic_with_no_shuffle(self, tmp_path):
        """Without shuffle, compaction is deterministic."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        self._write_seg_shard(shard_dir, n=10, strategy="sdf", quality=0.9)

        canon1 = tmp_path / "c1.npz"
        canon2 = tmp_path / "c2.npz"
        compact(shards_dir=shard_dir, output_path=canon1,
                epochs=1, shuffle=False)
        compact(shards_dir=shard_dir, output_path=canon2,
                epochs=1, shuffle=False)

        d1 = np.load(str(canon1))
        d2 = np.load(str(canon2))
        np.testing.assert_array_equal(d1["seg_W"], d2["seg_W"])
        np.testing.assert_array_equal(d1["seg_b"], d2["seg_b"])


# ── Integration ─────────────────────────────────────────────────────────

class TestFederationIntegration:
    def test_full_workflow(self, tmp_path):
        """Simulate: train → save shard → compact → load canonical."""
        clear_session_experiences()
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        canon = tmp_path / "canonical.npz"

        # 1. Train: selector learns that sdf is good for spheres
        sel = SegmentationStrategySelector()
        sphere_feats = _make_sphere_features()
        for _ in range(15):
            record_segmentation_experience(sphere_feats, "sdf", 0.95)
            feat_vec = np.array(sphere_feats, dtype=np.float64)
            sel._update_numpy(
                feat_vec, STRATEGY_NAMES.index("sdf"), 0.95)

        # 2. Save shard
        save_shard(shard_dir / "session1.jsonl")

        # 3. Compact
        stats = compact(shards_dir=shard_dir, output_path=canon, epochs=3)
        assert stats["status"] == "ok"

        # 4. Load into fresh selector
        sel2 = SegmentationStrategySelector()
        load_canonical(seg_selector=sel2, path=canon)

        # The loaded selector should prefer sdf for sphere-like features
        W, b = sel2.get_weights()
        sdf_idx = STRATEGY_NAMES.index("sdf")
        feat_vec = np.array(sphere_feats, dtype=np.float64)
        logits = W @ feat_vec + b
        assert logits[sdf_idx] > logits.mean()
