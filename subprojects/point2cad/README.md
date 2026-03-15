# Point2CAD Subproject

Wrapper for `Point2CAD` using either:

- the official Docker image `toshas/point2cad:v1`, or
- a local checkout of the official repo

## What it does

- samples a point cloud from a mesh
- generates rough surface labels from mesh face-normal clustering
- writes the `(x, y, z, s)` file expected by `Point2CAD`
- runs the official Point2CAD pipeline

## Why this one is different

`Point2CAD` reconstructs primitive surfaces, clipped surfaces, and topology. It is not a full CAD-program decoder like `CAD-Recode` or `cadrille`, but it is still a strong open local reverse-engineering stage.

## Single command

```bash
./subprojects/point2cad/launch.sh dev_models/beholder.stl runs/point2cad_beholder
```

This creates a small wrapper venv, pulls the pinned official `Point2CAD` image, generates the `(x, y, z, s)` point cloud, and runs the official Dockerized pipeline.

## Optional local repo path

```bash
./external/bootstrap_repos.sh

python3 subprojects/point2cad/run_point2cad.py \
  --mesh dev_models/beholder.stl \
  --out-dir runs/point2cad_beholder \
  --engine local \
  --point2cad-home external/point2cad
```
