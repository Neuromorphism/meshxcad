# cadrille Subproject

Wrapper for the official `cadrille` repository in point-cloud mode.

## What it does

- normalizes an input mesh into the `deepcad_test_mesh` format expected by `cadrille`
- runs official `test.py --split deepcad_test_mesh --mode pc`
- copies the predicted CadQuery program into the run directory
- exports STEP and STL from the generated code

## Single command

```bash
./subprojects/cadrille/launch.sh dev_models/beholder.stl runs/cadrille_beholder
```

This builds a pinned Docker image and runs the official `cadrille` point-cloud inference path inside it.

## Optional upstream checkout

```bash
./external/bootstrap_repos.sh
```

## Default checkpoint

- `maksimko123/cadrille-rl`

Use `--checkpoint-path maksimko123/cadrille` if you want the SFT checkpoint instead.
