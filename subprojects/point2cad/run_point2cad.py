#!/usr/bin/env python3
"""Prepare segmented point clouds from a mesh and run Point2CAD."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subprojects.common.mesh_preprocess import (
    label_sampled_faces,
    load_mesh,
    normalize_mesh,
    sample_point_cloud,
    save_mesh,
    save_metadata,
    save_point_cloud_xyzc,
)


def resolve_local_upstream(path_arg: str | None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(path_arg) if path_arg else None,
        Path(os.environ["POINT2CAD_HOME"]) if "POINT2CAD_HOME" in os.environ else None,
        repo_root / "external/point2cad",
    ]
    for candidate in candidates:
        if candidate and (candidate / "point2cad" / "main.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find Point2CAD checkout. Pass --point2cad-home or set POINT2CAD_HOME."
    )


def run_local(upstream: Path, xyzc_path: Path, out_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "point2cad.main",
        "--path_in",
        str(xyzc_path),
        "--path_out",
        str(out_dir),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{upstream}:{existing}" if existing else str(upstream)
    subprocess.run(command, cwd=upstream, env=env, check=True)


def run_docker(repo_root: Path, xyzc_path: Path, out_dir: Path) -> None:
    mount_root = repo_root.resolve()
    if mount_root not in xyzc_path.resolve().parents or mount_root not in out_dir.resolve().parents:
        raise ValueError("Docker mode requires --out-dir to stay under the repository root")
    input_rel = xyzc_path.resolve().relative_to(mount_root)
    output_rel = out_dir.resolve().relative_to(mount_root)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{mount_root}:/work/point2cad",
        "toshas/point2cad:v1",
        "python",
        "-m",
        "point2cad.main",
        "--path_in",
        f"/work/point2cad/{input_rel}",
        "--path_out",
        f"/work/point2cad/{output_rel}",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Point2CAD from a mesh")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--engine", choices=["docker", "local"], default="docker")
    parser.add_argument("--point2cad-home")
    parser.add_argument("--n-points", type=int, default=8192)
    parser.add_argument("--n-pre-points", type=int, default=32768)
    parser.add_argument("--surface-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)
    mesh, normalization = normalize_mesh(mesh, target="raw")
    save_mesh(mesh, out_dir / "input_mesh.stl")

    points, sampled_faces = sample_point_cloud(
        mesh,
        n_points=args.n_points,
        n_pre_points=args.n_pre_points,
        seed=args.seed,
    )
    labels = label_sampled_faces(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        sampled_faces,
        n_clusters=args.surface_clusters,
    )
    xyzc_path = save_point_cloud_xyzc(points, labels, out_dir / "input.xyzc")

    if args.engine == "docker":
        run_docker(repo_root, xyzc_path, out_dir)
        upstream = "docker:toshas/point2cad:v1"
    else:
        upstream_path = resolve_local_upstream(args.point2cad_home)
        run_local(upstream_path, xyzc_path, out_dir)
        upstream = str(upstream_path)

    save_metadata(
        out_dir / "run.json",
        method="point2cad",
        engine=args.engine,
        upstream=upstream,
        input_mesh=args.mesh,
        normalization=normalization,
        point_cloud=str(xyzc_path),
    )


if __name__ == "__main__":
    main()
