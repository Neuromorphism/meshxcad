# Mesh-to-CAD Subprojects

Each subproject accepts a mesh input and prepares the method-specific data expected by the upstream model.

## Included methods

- `subprojects/cad_recode`: direct local inference for `filapro/cad-recode-v1.5`
- `subprojects/cadrille`: wrapper around the official `cadrille` repo in point-cloud mode
- `subprojects/point2cad`: wrapper around the official `Point2CAD` repo or Docker image

## Shared behavior

- Input meshes are normalized consistently before inference.
- Point-cloud methods sample points directly from triangle meshes.
- Outputs are written into per-run directories under the chosen `--out-dir`.

## Typical usage

```bash
./subprojects/cad_recode/launch.sh dev_models/beholder.stl runs/cad_recode_beholder

./subprojects/cadrille/launch.sh dev_models/beholder.stl runs/cadrille_beholder

./subprojects/point2cad/launch.sh dev_models/beholder.stl runs/point2cad_beholder
```

For pinned upstream checkouts, run:

```bash
./external/bootstrap_repos.sh
```
