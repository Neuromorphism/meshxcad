# External Bootstrap

This directory is for pinned upstream checkouts used by the mesh-to-CAD subprojects.

Tracked files here are only bootstrap metadata and scripts. The actual cloned repos are ignored by git.

## Bootstrap everything

```bash
./external/bootstrap_repos.sh
```

That script clones and pins:

- `external/cad-recode`
- `external/cadrille`
- `external/point2cad`

The pinned refs live in [pins.env](/home/me/gits/meshxcad/external/pins.env).
