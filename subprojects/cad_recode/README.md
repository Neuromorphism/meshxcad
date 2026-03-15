# CAD-Recode Subproject

Local runner for `filapro/cad-recode-v1.5`.

## What it does

- loads a mesh
- normalizes it the same way as the official demo
- samples a 256-point cloud
- runs the `CAD-Recode` checkpoint locally
- saves:
  - generated CadQuery program
  - exported STEP
  - exported STL
  - sampled point cloud and metadata

## Single command

```bash
./subprojects/cad_recode/launch.sh dev_models/beholder.stl runs/cad_recode_beholder
```

By default this uses a pinned local venv under `.venvs/cad-recode`.

Set `CAD_RECODE_ENGINE=docker` to run the same subproject in Docker.

## Manual bootstrap

```bash
./subprojects/cad_recode/bootstrap_venv.sh
```

## Notes

- Default checkpoint: `filapro/cad-recode-v1.5`
- The script prefers CUDA and will use `flash_attention_2` when available.
- This path is the cleanest fully local runner in this repo because the official project publishes the core model logic in the notebook.
