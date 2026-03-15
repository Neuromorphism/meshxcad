#!/usr/bin/env python3
"""Run the official cadrille point-cloud inference on a custom mesh."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cadquery as cq
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subprojects.common.mesh_preprocess import load_mesh, normalize_mesh, save_metadata, save_mesh


def resolve_upstream(path_arg: str | None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(path_arg) if path_arg else None,
        Path(os.environ["CADRILLE_HOME"]) if "CADRILLE_HOME" in os.environ else None,
        repo_root / "external/cadrille",
    ]
    for candidate in candidates:
        if candidate and (candidate / "test.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find cadrille checkout. Pass --cadrille-home or set CADRILLE_HOME."
    )


def export_program(program_path: Path, out_dir: Path) -> tuple[Path, Path]:
    namespace = {"cq": cq}
    exec(program_path.read_text(), namespace)
    if "r" not in namespace:
        raise RuntimeError("Generated program did not define the expected CadQuery result variable 'r'")
    compound = namespace["r"].val()
    step_path = out_dir / "model.step"
    stl_path = out_dir / "model.stl"
    cq.exporters.export(compound, step_path)
    vertices, faces = compound.tessellate(0.001, 0.1)
    save_mesh(trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces), stl_path)
    return step_path, stl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cadrille point-cloud inference from a mesh")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cadrille-home")
    parser.add_argument("--checkpoint-path", default="maksimko123/cadrille-rl")
    args = parser.parse_args()

    upstream = resolve_upstream(args.cadrille_home)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)
    mesh, normalization = normalize_mesh(mesh, target="unit_cube")

    data_root = out_dir / "data"
    split_dir = data_root / "deepcad_test_mesh"
    split_dir.mkdir(parents=True, exist_ok=True)
    normalized_mesh_path = split_dir / "input.stl"
    save_mesh(mesh, normalized_mesh_path)

    py_dir = out_dir / "generated_py"
    if py_dir.exists():
        shutil.rmtree(py_dir)
    py_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "test.py",
        "--data-path",
        str(data_root),
        "--split",
        "deepcad_test_mesh",
        "--mode",
        "pc",
        "--checkpoint-path",
        args.checkpoint_path,
        "--py-path",
        str(py_dir),
    ]
    subprocess.run(command, cwd=upstream, check=True)

    generated = sorted(py_dir.glob("*.py"))
    if not generated:
        raise RuntimeError("cadrille did not produce any CadQuery program")

    program_path = out_dir / "program.py"
    shutil.copy2(generated[0], program_path)
    step_path, stl_path = export_program(program_path, out_dir)

    save_metadata(
        out_dir / "run.json",
        method="cadrille",
        checkpoint_path=args.checkpoint_path,
        upstream=str(upstream),
        input_mesh=args.mesh,
        normalization=normalization,
        outputs={
            "program": program_path,
            "step": step_path,
            "stl": stl_path,
        },
    )


if __name__ == "__main__":
    main()
