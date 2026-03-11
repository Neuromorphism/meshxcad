"""Integration tests that require FreeCAD.

These tests generate actual FreeCAD documents, tessellate them, and verify
the full pipeline of detail transfer through CAD I/O.

Skipped automatically if FreeCAD is not available.
"""

import os
import tempfile
import numpy as np
import pytest

try:
    import FreeCAD
    import Part
    import Mesh
    import MeshPart
    HAS_FREECAD = True
except ImportError:
    HAS_FREECAD = False

pytestmark = pytest.mark.skipif(not HAS_FREECAD, reason="FreeCAD not available")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestTestDataGeneration:
    """Test that test data generators produce valid files."""

    def test_cube_set_generation(self, temp_dir):
        from meshxcad.test_data_gen import generate_cube_set
        paths = generate_cube_set(temp_dir)

        for key in ["plain_mesh", "featured_mesh", "plain_cad", "featured_cad"]:
            assert key in paths
            assert os.path.exists(paths[key]), f"{key} file not created"
            assert os.path.getsize(paths[key]) > 0, f"{key} file is empty"

    def test_sphere_set_generation(self, temp_dir):
        from meshxcad.test_data_gen import generate_sphere_set
        paths = generate_sphere_set(temp_dir)

        for key in ["plain_mesh", "featured_mesh", "plain_cad", "featured_cad"]:
            assert os.path.exists(paths[key])

    def test_cylinder_set_generation(self, temp_dir):
        from meshxcad.test_data_gen import generate_cylinder_set
        paths = generate_cylinder_set(temp_dir)

        for key in ["plain_mesh", "featured_mesh", "plain_cad", "featured_cad"]:
            assert os.path.exists(paths[key])


class TestCADIO:
    """Test CAD I/O operations."""

    def test_load_and_tessellate(self, temp_dir):
        """Create a CAD doc, save, reload, and tessellate."""
        from meshxcad import cad_io, mesh_io

        # Create a simple box
        doc = FreeCAD.newDocument("Test")
        box = doc.addObject("Part::Box", "Box")
        box.Length = 10
        box.Width = 10
        box.Height = 10
        doc.recompute()

        path = os.path.join(temp_dir, "test.FCStd")
        doc.saveAs(path)
        FreeCAD.closeDocument(doc.Name)

        # Reload and tessellate
        doc2 = cad_io.load_cad(path)
        shape = cad_io.cad_to_shape(doc2)
        mesh = cad_io.shape_to_mesh(shape)
        verts, faces = mesh_io.mesh_to_numpy(mesh)
        cad_io.close_document(doc2)

        assert len(verts) > 0
        assert len(faces) > 0
        # Box should have 6 faces, but tessellated into many triangles
        assert len(faces) >= 12


class TestFullPipelineCube:
    """Full pipeline test: plain CAD ← featured mesh → featured CAD."""

    def test_mesh_modifies_cad(self, temp_dir):
        """Modify a plain cube CAD using a featured cube mesh."""
        from meshxcad.test_data_gen import generate_cube_set
        from meshxcad.mesh_to_cad import transfer_detail

        paths = generate_cube_set(temp_dir)
        output = os.path.join(temp_dir, "result.stl")

        transfer_detail(
            paths["plain_cad"],
            paths["featured_mesh"],
            output,
        )

        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

        # Load result and the objective (featured mesh)
        from meshxcad import mesh_io
        result_mesh = mesh_io.load_mesh(output)
        result_v, result_f = mesh_io.mesh_to_numpy(result_mesh)

        objective_mesh = mesh_io.load_mesh(paths["featured_mesh"])
        obj_v, obj_f = mesh_io.mesh_to_numpy(objective_mesh)

        # Result should resemble the featured mesh
        from meshxcad.alignment import find_correspondences
        _, _, dists = find_correspondences(result_v, obj_v)
        assert np.mean(dists) < 1.0, f"Mean distance to objective: {np.mean(dists)}"

    def test_cad_modifies_mesh(self, temp_dir):
        """Modify a plain cube mesh using a featured cube CAD."""
        from meshxcad.test_data_gen import generate_cube_set
        from meshxcad.cad_to_mesh import transfer_detail

        paths = generate_cube_set(temp_dir)
        output = os.path.join(temp_dir, "result.stl")

        transfer_detail(
            paths["plain_mesh"],
            paths["featured_cad"],
            output,
        )

        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

        from meshxcad import mesh_io
        result_mesh = mesh_io.load_mesh(output)
        result_v, result_f = mesh_io.mesh_to_numpy(result_mesh)

        objective_mesh = mesh_io.load_mesh(paths["featured_mesh"])
        obj_v, obj_f = mesh_io.mesh_to_numpy(objective_mesh)

        from meshxcad.alignment import find_correspondences
        _, _, dists = find_correspondences(result_v, obj_v)
        assert np.mean(dists) < 1.0
