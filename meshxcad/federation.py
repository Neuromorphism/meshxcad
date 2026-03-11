"""Federated learning for selectors and refiners.

Enables multiple remote sessions to contribute training experiences that
get merged back into the repository's learned weights without conflicts.

Architecture
------------
Instead of merging weight matrices (which is lossy and order-dependent),
we use an **experience replay log** approach:

1. **Experience Shards** — each training session appends observations
   (features, action chosen, reward received) to a JSONL shard file
   under ``learned_weights/shards/``.  Each session writes its own file,
   so there are never merge conflicts.

2. **Canonical Weights** — a ``.npz`` checkpoint under
   ``learned_weights/canonical.npz`` representing the current best
   weights, compiled by replaying all experience shards.

3. **Compaction** — replays every shard through fresh selectors (using
   the REINFORCE / reward gradient updates already in ``optim.py``) to
   produce a new canonical checkpoint.  Old shards can optionally be
   archived after compaction.

Workflow
--------
Remote session::

    from meshxcad.federation import auto_save_shard
    # ... run training loops ...
    auto_save_shard()  # writes learned_weights/shards/<session>_<ts>.jsonl
    # git add + commit + push

CI/CD or local::

    python -m meshxcad.federation compact
    # Replays all shards → learned_weights/canonical.npz
    # Commits the updated canonical file

Startup::

    from meshxcad.federation import load_canonical
    load_canonical()  # loads weights into the global selectors
"""

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

# Repo-relative paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = _REPO_ROOT / "learned_weights"
SHARDS_DIR = WEIGHTS_DIR / "shards"
CANONICAL_PATH = WEIGHTS_DIR / "canonical.npz"

# In-memory session experience buffer
_session_id = uuid.uuid4().hex[:12]
_session_experiences = []  # list of dicts


# ── Experience recording ────────────────────────────────────────────────

def record_experience(kind, **kwargs):
    """Record a single training experience for later export.

    Args:
        kind: "segmentation" or "fixer"
        **kwargs: experience-specific fields
    """
    entry = {
        "kind": kind,
        "session": _session_id,
        "timestamp": time.time(),
    }
    entry.update(kwargs)
    _session_experiences.append(entry)


def record_segmentation_experience(features, strategy, quality):
    """Record a segmentation strategy selection experience.

    Args:
        features: feature vector (list or ndarray of 6 floats)
        strategy: strategy name chosen
        quality: 0–1 quality score observed
    """
    record_experience(
        "segmentation",
        features=list(np.asarray(features, dtype=float)),
        strategy=str(strategy),
        quality=float(quality),
    )


def record_fixer_experience(fixer_names, chosen, diff_scores,
                            tried, fixer_history, improved,
                            improvement_amount):
    """Record a fixer selection experience.

    Args:
        fixer_names: ordered list of all fixer names
        chosen: name of the fixer that was applied
        diff_scores: dict {name: score}
        tried: set/list of already-tried names
        fixer_history: dict {name: {"avg_improvement": float}}
        improved: bool
        improvement_amount: float
    """
    record_experience(
        "fixer",
        fixer_names=list(fixer_names),
        chosen=str(chosen),
        diff_scores={str(k): float(v) for k, v in diff_scores.items()},
        tried=list(tried),
        fixer_history={
            str(k): {"avg_improvement": float(v.get("avg_improvement", 50.0))}
            for k, v in fixer_history.items()
        },
        improved=bool(improved),
        improvement_amount=float(improvement_amount),
    )


def get_session_experiences():
    """Return list of experiences recorded this session."""
    return list(_session_experiences)


def clear_session_experiences():
    """Clear the in-memory experience buffer."""
    _session_experiences.clear()


# ── Shard I/O ───────────────────────────────────────────────────────────

def save_shard(path=None):
    """Write current session experiences to a JSONL shard file.

    Args:
        path: explicit path, or None to auto-generate under shards/

    Returns:
        Path: the shard file that was written (or None if nothing to save)
    """
    if not _session_experiences:
        return None

    if path is None:
        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = SHARDS_DIR / f"{_session_id}_{ts}.jsonl"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for exp in _session_experiences:
            f.write(json.dumps(exp, separators=(",", ":")) + "\n")

    return path


def auto_save_shard():
    """Convenience: save shard and return its path for git add."""
    path = save_shard()
    if path:
        rel = path.relative_to(_REPO_ROOT)
        print(f"Saved {len(_session_experiences)} experiences to {rel}")
    return path


def load_shard(path):
    """Load experiences from a JSONL shard file.

    Returns:
        list of dicts
    """
    experiences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                experiences.append(json.loads(line))
    return experiences


def list_shards(shards_dir=None):
    """List all .jsonl shard files, sorted by name (oldest first).

    Returns:
        list of Path objects
    """
    d = Path(shards_dir) if shards_dir else SHARDS_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.jsonl"))


# ── Canonical weight I/O ────────────────────────────────────────────────

def save_canonical(seg_selector, fixer_selector=None, path=None):
    """Save current selector weights to the canonical checkpoint.

    Args:
        seg_selector: SegmentationStrategySelector instance
        fixer_selector: FixerPrioritySelector instance (optional)
        path: override path (default: learned_weights/canonical.npz)
    """
    path = Path(path) if path else CANONICAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {}

    # Segmentation weights
    W, b = seg_selector.get_weights()
    data["seg_W"] = np.asarray(W)
    data["seg_b"] = np.asarray(b)

    # Fixer weights (variable size, depends on fixer list)
    if fixer_selector is not None:
        if hasattr(fixer_selector, "_w"):
            from .optim import _HAS_TORCH
            if _HAS_TORCH:
                import torch
                w = fixer_selector._w.detach().numpy().copy()
                bias = fixer_selector._bias.detach().numpy().copy()
            else:
                w = fixer_selector._w.copy()
                bias = fixer_selector._bias.copy()
            data["fixer_w"] = w
            data["fixer_bias"] = bias
            # Store fixer names for reconstruction
            data["fixer_names"] = np.array(
                fixer_selector.fixer_names, dtype=object)

    np.savez(str(path), **data)


def load_canonical(seg_selector=None, fixer_selector=None, path=None):
    """Load canonical weights into selector instances.

    If selectors are None, loads into the global singletons from
    segmentation.py and adversarial_loop.py.

    Args:
        seg_selector: SegmentationStrategySelector to populate
        fixer_selector: FixerPrioritySelector to populate
        path: override path

    Returns:
        bool: True if loaded successfully
    """
    path = Path(path) if path else CANONICAL_PATH
    if not path.exists():
        return False

    data = np.load(str(path), allow_pickle=True)

    # Load segmentation weights
    if seg_selector is None:
        try:
            from .segmentation import get_strategy_selector
            seg_selector = get_strategy_selector()
        except (ImportError, AttributeError):
            seg_selector = None

    if seg_selector is not None and "seg_W" in data:
        seg_selector.set_weights(data["seg_W"], data["seg_b"])

    # Load fixer weights
    if fixer_selector is None:
        try:
            from .adversarial_loop import get_fixer_selector
            fixer_selector = get_fixer_selector()
        except (ImportError, AttributeError):
            fixer_selector = None

    if fixer_selector is not None and "fixer_w" in data:
        from .optim import _HAS_TORCH
        w = data["fixer_w"]
        bias = data["fixer_bias"]
        if _HAS_TORCH:
            import torch
            with torch.no_grad():
                fixer_selector._w.copy_(
                    torch.tensor(w, dtype=torch.float64))
                # Bias may have different length if fixer list changed
                n = min(len(bias), len(fixer_selector._bias))
                fixer_selector._bias[:n] = torch.tensor(
                    bias[:n], dtype=torch.float64)
        else:
            fixer_selector._w = np.array(w, dtype=np.float64)
            n = min(len(bias), len(fixer_selector._bias))
            fixer_selector._bias[:n] = bias[:n]

    return True


# ── Compaction ──────────────────────────────────────────────────────────

def compact(shards_dir=None, output_path=None, lr=None,
            epochs=1, shuffle=True, archive=False):
    """Replay all experience shards to produce compacted canonical weights.

    This is the core federated learning merge: rather than averaging
    weight matrices (which is mathematically dubious), we replay every
    recorded experience through a fresh selector, applying the same
    REINFORCE / reward-gradient updates that the original sessions used.
    Multiple epochs and shuffling help the model see all experiences in
    varied orders, reducing recency bias.

    Args:
        shards_dir: directory containing .jsonl shards
        output_path: where to write the canonical .npz
        lr: learning rate override (None = use selector defaults)
        epochs: how many times to replay all experiences
        shuffle: randomize experience order within each epoch
        archive: if True, move processed shards to an archive/ subdir

    Returns:
        dict with compaction statistics
    """
    from .optim import (
        SegmentationStrategySelector,
        FixerPrioritySelector,
        _features_to_vector,
        STRATEGY_NAMES,
    )

    shards = list_shards(shards_dir)
    if not shards:
        return {"status": "no_shards", "experiences": 0}

    # Collect all experiences
    all_experiences = []
    for shard_path in shards:
        all_experiences.extend(load_shard(shard_path))

    if not all_experiences:
        return {"status": "empty_shards", "experiences": 0}

    # Separate by type
    seg_exps = [e for e in all_experiences if e["kind"] == "segmentation"]
    fixer_exps = [e for e in all_experiences if e["kind"] == "fixer"]

    # Sort by timestamp for deterministic base ordering
    seg_exps.sort(key=lambda e: e.get("timestamp", 0))
    fixer_exps.sort(key=lambda e: e.get("timestamp", 0))

    stats = {
        "total_experiences": len(all_experiences),
        "segmentation_experiences": len(seg_exps),
        "fixer_experiences": len(fixer_exps),
        "shards_processed": len(shards),
        "epochs": epochs,
    }

    # ── Replay segmentation experiences ──
    seg_sel = SegmentationStrategySelector(lr=lr or 0.05)
    # Start from canonical if it exists (incremental compaction)
    canon = Path(output_path) if output_path else CANONICAL_PATH
    if canon.exists():
        try:
            data = np.load(str(canon), allow_pickle=True)
            if "seg_W" in data:
                seg_sel.set_weights(data["seg_W"], data["seg_b"])
        except Exception:
            pass  # start fresh

    rng = np.random.RandomState(42)
    for epoch in range(epochs):
        order = list(range(len(seg_exps)))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            exp = seg_exps[i]
            feat_vec = np.array(exp["features"], dtype=np.float64)
            strategy_idx = STRATEGY_NAMES.index(exp["strategy"])
            quality = exp["quality"]
            # Direct weight update (bypass mesh_features since we have
            # the pre-computed feature vector)
            seg_sel._history.append(
                (feat_vec.copy(), strategy_idx, quality))
            from .optim import _HAS_TORCH
            if _HAS_TORCH:
                seg_sel._update_torch(feat_vec, strategy_idx, quality)
            else:
                seg_sel._update_numpy(feat_vec, strategy_idx, quality)

    # ── Replay fixer experiences ──
    fixer_sel = None
    if fixer_exps:
        # Collect the union of all fixer names seen across shards
        all_fixer_names = set()
        for exp in fixer_exps:
            all_fixer_names.update(exp.get("fixer_names", []))
        all_fixer_names = sorted(all_fixer_names)

        fixer_sel = FixerPrioritySelector(
            all_fixer_names, lr=lr or 0.02)

        # Load existing canonical fixer weights if available
        if canon.exists():
            try:
                data = np.load(str(canon), allow_pickle=True)
                if "fixer_w" in data:
                    from .optim import _HAS_TORCH
                    w = data["fixer_w"]
                    if _HAS_TORCH:
                        import torch
                        with torch.no_grad():
                            fixer_sel._w.copy_(
                                torch.tensor(w, dtype=torch.float64))
                    else:
                        fixer_sel._w = np.array(w, dtype=np.float64)
            except Exception:
                pass

        for epoch in range(epochs):
            order = list(range(len(fixer_exps)))
            if shuffle:
                rng.shuffle(order)
            for i in order:
                exp = fixer_exps[i]
                fixer_sel.update(
                    chosen_name=exp["chosen"],
                    diff_scores=exp["diff_scores"],
                    tried_set=set(exp["tried"]),
                    fixer_history=exp["fixer_history"],
                    improved=exp["improved"],
                    improvement_amount=exp["improvement_amount"],
                )

    # ── Save canonical ──
    save_canonical(seg_sel, fixer_sel, output_path)
    stats["status"] = "ok"

    # ── Archive shards if requested ──
    if archive:
        archive_dir = (Path(shards_dir) if shards_dir
                       else SHARDS_DIR) / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for shard_path in shards:
            dest = archive_dir / shard_path.name
            shard_path.rename(dest)
        stats["archived"] = len(shards)

    return stats


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    """CLI for federation operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Federated learning for meshxcad selectors")
    sub = parser.add_subparsers(dest="command")

    # compact
    p_compact = sub.add_parser(
        "compact", help="Replay all shards into canonical weights")
    p_compact.add_argument(
        "--shards-dir", default=None,
        help="Directory containing .jsonl shards")
    p_compact.add_argument(
        "--output", default=None,
        help="Output path for canonical.npz")
    p_compact.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate override")
    p_compact.add_argument(
        "--epochs", type=int, default=3,
        help="Number of replay epochs (default: 3)")
    p_compact.add_argument(
        "--no-shuffle", action="store_true",
        help="Disable experience shuffling")
    p_compact.add_argument(
        "--archive", action="store_true",
        help="Move processed shards to archive/ after compaction")

    # save-shard (for manual use)
    p_save = sub.add_parser(
        "save-shard", help="Save current session experiences to a shard")

    # info
    p_info = sub.add_parser(
        "info", help="Show statistics about shards and canonical weights")

    args = parser.parse_args()

    if args.command == "compact":
        stats = compact(
            shards_dir=args.shards_dir,
            output_path=args.output,
            lr=args.lr,
            epochs=args.epochs,
            shuffle=not args.no_shuffle,
            archive=args.archive,
        )
        print(f"Compaction complete:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.command == "save-shard":
        path = auto_save_shard()
        if path is None:
            print("No experiences to save.")

    elif args.command == "info":
        _print_info()

    else:
        parser.print_help()


def _print_info():
    """Print statistics about shards and canonical weights."""
    shards = list_shards()
    print(f"Shards directory: {SHARDS_DIR}")
    print(f"  Shard files: {len(shards)}")

    total = 0
    seg_count = 0
    fixer_count = 0
    sessions = set()
    for s in shards:
        exps = load_shard(s)
        total += len(exps)
        for e in exps:
            if e["kind"] == "segmentation":
                seg_count += 1
            elif e["kind"] == "fixer":
                fixer_count += 1
            sessions.add(e.get("session", "?"))
    print(f"  Total experiences: {total}")
    print(f"    Segmentation: {seg_count}")
    print(f"    Fixer: {fixer_count}")
    print(f"  Unique sessions: {len(sessions)}")

    print(f"\nCanonical weights: {CANONICAL_PATH}")
    if CANONICAL_PATH.exists():
        data = np.load(str(CANONICAL_PATH), allow_pickle=True)
        print(f"  Keys: {list(data.keys())}")
        if "seg_W" in data:
            print(f"  seg_W shape: {data['seg_W'].shape}")
            print(f"  seg_b shape: {data['seg_b'].shape}")
        if "fixer_w" in data:
            print(f"  fixer_w shape: {data['fixer_w'].shape}")
            print(f"  fixer_bias shape: {data['fixer_bias'].shape}")
    else:
        print("  (not yet created — run compact)")


if __name__ == "__main__":
    main()
