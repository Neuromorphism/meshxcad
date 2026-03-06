# MeshXCAD

Bidirectional detail transfer between 3D meshes and parametric CAD models.

## Overview

MeshXCAD enables:
- **Mesh → CAD**: Add surface details from a mesh onto a plain parametric CAD body
- **CAD → Mesh**: Add geometric features from a CAD model onto a plain mesh

## Requirements

- Python 3.8+
- FreeCAD (with Python bindings)
- numpy
- scipy

## Usage

```python
from meshxcad import mesh_to_cad, cad_to_mesh

# Add mesh detail onto a plain CAD model
featured_cad = mesh_to_cad.transfer_detail("plain.FCStd", "featured.stl", "output.FCStd")

# Add CAD detail onto a plain mesh
featured_mesh = cad_to_mesh.transfer_detail("plain.stl", "featured.FCStd", "output.stl")
```

## Test Data

Each test object set contains four files:
1. `plain.stl` — plain mesh (e.g., a cube)
2. `featured.stl` — mesh with surface features
3. `plain.FCStd` — plain parametric CAD model
4. `featured.FCStd` — CAD model with features

## Running Tests

```bash
python -m pytest tests/ -v
```
