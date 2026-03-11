"""Tests for meshxcad.step_io — STEP/IGES reading and tessellation."""

import os
import tempfile

import numpy as np
import pytest

from meshxcad.step_io import read_step, _tessellate_shape, _deduplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step_box(path, lx=10.0, ly=10.0, lz=5.0):
    """Create a STEP file containing a simple box via cadquery."""
    import cadquery as cq
    result = cq.Workplane("XY").box(lx, ly, lz)
    cq.exporters.export(result, path, exportType="STEP")


def _make_step_cylinder(path, radius=5.0, height=10.0):
    """Create a STEP file containing a cylinder."""
    import cadquery as cq
    result = cq.Workplane("XY").circle(radius).extrude(height)
    cq.exporters.export(result, path, exportType="STEP")


def _make_step_flange(path):
    """Create a STEP file with a pipe flange (cylinder + holes)."""
    import cadquery as cq
    result = (
        cq.Workplane("XY")
        .circle(10)
        .extrude(3)
        .faces(">Z")
        .workplane()
        .circle(5)
        .cutThruAll()
        .faces(">Z")
        .workplane()
        .polarArray(7.5, 0, 360, 6)
        .circle(1.2)
        .cutThruAll()
    )
    cq.exporters.export(result, path, exportType="STEP")


def _make_step_sphere(path, radius=5.0):
    """Create a STEP file containing a sphere."""
    import cadquery as cq
    result = cq.Workplane("XY").sphere(radius)
    cq.exporters.export(result, path, exportType="STEP")


def _make_step_assembly(path):
    """Create a multi-body STEP file."""
    import cadquery as cq
    box = cq.Workplane("XY").box(10, 10, 5)
    cyl = cq.Workplane("XY").transformed(offset=(0, 0, 5)).circle(3).extrude(8)
    result = box.union(cyl)
    cq.exporters.export(result, path, exportType="STEP")


# ---------------------------------------------------------------------------
# Tests: read_step
# ---------------------------------------------------------------------------

class TestReadStep:

    def test_box(self, tmp_path):
        path = str(tmp_path / "box.step")
        _make_step_box(path)
        v, f = read_step(path)
        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 0
        assert len(f) > 0
        # Box should have reasonable extents
        bbox = v.max(axis=0) - v.min(axis=0)
        assert bbox[0] == pytest.approx(10.0, abs=0.5)
        assert bbox[1] == pytest.approx(10.0, abs=0.5)
        assert bbox[2] == pytest.approx(5.0, abs=0.5)

    def test_cylinder(self, tmp_path):
        path = str(tmp_path / "cyl.step")
        _make_step_cylinder(path, radius=5.0, height=10.0)
        v, f = read_step(path)
        assert len(v) >= 10
        assert len(f) >= 10
        bbox = v.max(axis=0) - v.min(axis=0)
        # Diameter ~10, height 10
        assert bbox[0] == pytest.approx(10.0, abs=1.0)
        assert bbox[2] == pytest.approx(10.0, abs=1.0)

    def test_sphere(self, tmp_path):
        path = str(tmp_path / "sphere.step")
        _make_step_sphere(path, radius=5.0)
        v, f = read_step(path)
        assert len(v) >= 20
        # Diameter ~10 on all axes
        bbox = v.max(axis=0) - v.min(axis=0)
        for d in range(3):
            assert bbox[d] == pytest.approx(10.0, abs=1.0)

    def test_flange_with_holes(self, tmp_path):
        path = str(tmp_path / "flange.step")
        _make_step_flange(path)
        v, f = read_step(path)
        assert len(v) > 100  # complex shape → many vertices
        assert len(f) > 100

    def test_assembly(self, tmp_path):
        path = str(tmp_path / "assembly.step")
        _make_step_assembly(path)
        v, f = read_step(path)
        assert len(v) > 50
        bbox = v.max(axis=0) - v.min(axis=0)
        # Box 10x10x5 + cylinder up to z=13 → z range ~15.5
        assert bbox[2] > 10

    def test_stp_extension(self, tmp_path):
        path = str(tmp_path / "model.stp")
        _make_step_box(path)
        v, f = read_step(path)
        assert len(v) > 0

    def test_custom_deflection(self, tmp_path):
        path = str(tmp_path / "sphere.step")
        _make_step_sphere(path)
        # Finer tessellation → more vertices
        v_coarse, _ = read_step(path, linear_deflection=1.0)
        v_fine, _ = read_step(path, linear_deflection=0.05)
        assert len(v_fine) > len(v_coarse)

    def test_nonexistent_file(self):
        with pytest.raises(IOError):
            read_step("/tmp/nonexistent_file.step")

    def test_unsupported_extension(self, tmp_path):
        path = str(tmp_path / "model.dwg")
        with open(path, "w") as f:
            f.write("dummy")
        with pytest.raises(ValueError, match="Unsupported CAD format"):
            read_step(path)

    def test_vertex_dtype(self, tmp_path):
        path = str(tmp_path / "box.step")
        _make_step_box(path)
        v, f = read_step(path)
        assert v.dtype == np.float64
        assert f.dtype == np.int64

    def test_face_indices_valid(self, tmp_path):
        path = str(tmp_path / "box.step")
        _make_step_box(path)
        v, f = read_step(path)
        assert f.min() >= 0
        assert f.max() < len(v)


# ---------------------------------------------------------------------------
# Tests: _deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:

    def test_no_duplicates(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int64)
        v2, f2 = _deduplicate(v, f)
        assert len(v2) == 3
        # np.unique may reorder vertices, so check that face still
        # references the same 3 distinct coordinates
        tri_coords = set(map(tuple, v2[f2[0]]))
        expected = {(0, 0, 0), (1, 0, 0), (0, 1, 0)}
        assert tri_coords == expected

    def test_with_duplicates(self):
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 0], [1, 0, 0],  # duplicates of 0 and 1
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 2]], dtype=np.int64)
        v2, f2 = _deduplicate(v, f)
        assert len(v2) == 3
        # Both faces should reference the same vertices
        assert f2[0, 0] == f2[1, 0]  # both point to [0,0,0]

    def test_near_duplicates(self):
        v = np.array([
            [0, 0, 0],
            [1e-10, 0, 0],  # near-duplicate within tol=1e-8
        ], dtype=np.float64)
        f = np.array([[0, 1, 0]], dtype=np.int64)
        v2, f2 = _deduplicate(v, f, tol=1e-8)
        assert len(v2) == 1


# ---------------------------------------------------------------------------
# Tests: CLI integration with STEP files
# ---------------------------------------------------------------------------

class TestCLIStepIntegration:

    def test_step_as_target(self, tmp_path):
        """STEP file used as target mesh input."""
        step_path = str(tmp_path / "target.step")
        _make_step_box(step_path, lx=10.0, ly=10.0, lz=5.0)

        from meshxcad.__main__ import load_mesh, optimise
        # Use finer tessellation to get enough vertices for accuracy
        from meshxcad.step_io import read_step
        v, f = read_step(step_path, linear_deflection=0.05)
        assert len(v) > 0
        result = optimise(v, f, max_sweeps=1, rounds=2, verbose=False)
        assert result["_program_obj"] is not None
        assert "final" in result

    def test_step_as_cad_input(self, tmp_path):
        """STEP file used as starting CAD (-c flag)."""
        step_path = str(tmp_path / "cad.step")
        _make_step_cylinder(step_path)

        # Target: a different mesh (sphere)
        from meshxcad.synthetic import make_sphere_mesh
        target_v, target_f = make_sphere_mesh(radius=5.0)

        from meshxcad.__main__ import _load_cad_from_step, optimise
        prog = _load_cad_from_step(step_path, target_v, target_f, quiet=True)
        assert prog.n_enabled() >= 1

        result = optimise(target_v, target_f, initial_cad=prog,
                          max_sweeps=1, rounds=2, verbose=False)
        assert result["final"]["accuracy"] > 0

    def test_step_flange_target(self, tmp_path):
        """Complex STEP (flange with holes) as target."""
        step_path = str(tmp_path / "flange.step")
        _make_step_flange(step_path)

        from meshxcad.__main__ import load_mesh, optimise
        v, f = load_mesh(step_path)
        assert len(v) > 100

        result = optimise(v, f, max_sweeps=2, rounds=3, verbose=False)
        assert result["final"]["n_ops"] >= 1
        assert result["converged"] or result["total_sweeps"] == 2

    def test_step_output_files(self, tmp_path):
        """Verify output files are created."""
        step_path = str(tmp_path / "box.step")
        _make_step_box(step_path)
        out_dir = str(tmp_path / "out")

        from meshxcad.__main__ import load_mesh, optimise
        import json
        v, f = load_mesh(step_path)
        result = optimise(v, f, max_sweeps=1, rounds=1, verbose=False)

        # Write outputs manually (like CLI does)
        os.makedirs(out_dir, exist_ok=True)
        prog_path = os.path.join(out_dir, "program.json")
        with open(prog_path, "w") as fp:
            json.dump({"program": result["program"]}, fp)

        from meshxcad.stl_io import write_binary_stl
        cad_v, cad_f = result["_program_obj"].evaluate()
        stl_path = os.path.join(out_dir, "output.stl")
        write_binary_stl(stl_path, cad_v, cad_f)

        assert os.path.isfile(prog_path)
        assert os.path.isfile(stl_path)
        assert os.path.getsize(stl_path) > 0

    def test_load_mesh_dispatches_step(self, tmp_path):
        """load_mesh correctly dispatches .step extension."""
        path = str(tmp_path / "test.step")
        _make_step_box(path)
        from meshxcad.__main__ import load_mesh
        v, f = load_mesh(path)
        assert v.shape[1] == 3
        assert f.shape[1] == 3

    def test_load_mesh_dispatches_stp(self, tmp_path):
        """load_mesh correctly dispatches .stp extension."""
        path = str(tmp_path / "test.stp")
        _make_step_box(path)
        from meshxcad.__main__ import load_mesh
        v, f = load_mesh(path)
        assert len(v) > 0

    def test_is_step_file(self):
        from meshxcad.__main__ import _is_step_file
        assert _is_step_file("model.step")
        assert _is_step_file("model.stp")
        assert _is_step_file("model.STEP")
        assert _is_step_file("model.iges")
        assert _is_step_file("model.igs")
        assert not _is_step_file("model.stl")
        assert not _is_step_file("model.json")
        assert not _is_step_file("model.obj")


# ---------------------------------------------------------------------------
# Tests: OBJ and PLY readers (from __main__)
# ---------------------------------------------------------------------------

class TestMeshReaders:

    def test_obj_reader(self, tmp_path):
        """OBJ reader handles basic triangles."""
        path = str(tmp_path / "cube.obj")
        with open(path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
            f.write("v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n")
            f.write("f 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\n")
        from meshxcad.__main__ import load_mesh
        v, f = load_mesh(path)
        assert len(v) == 8
        assert len(f) == 4

    def test_obj_reader_quads(self, tmp_path):
        """OBJ reader triangulates quads."""
        path = str(tmp_path / "quad.obj")
        with open(path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
            f.write("f 1 2 3 4\n")  # quad → 2 triangles
        from meshxcad.__main__ import load_mesh
        v, f = load_mesh(path)
        assert len(v) == 4
        assert len(f) == 2

    def test_obj_reader_with_normals(self, tmp_path):
        """OBJ reader handles v/vt/vn face format."""
        path = str(tmp_path / "normals.obj")
        with open(path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
            f.write("vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n")
            f.write("f 1//1 2//2 3//3\n")
        from meshxcad.__main__ import load_mesh
        v, f = load_mesh(path)
        assert len(v) == 3
        assert len(f) == 1

    def test_ply_ascii_reader(self, tmp_path):
        """PLY ASCII reader."""
        path = str(tmp_path / "tri.ply")
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex 3\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("element face 1\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.write("0 0 0\n1 0 0\n0 1 0\n")
            f.write("3 0 1 2\n")
        from meshxcad.__main__ import load_mesh
        v, f = load_mesh(path)
        assert len(v) == 3
        assert len(f) == 1

    def test_unsupported_format(self, tmp_path):
        path = str(tmp_path / "model.xyz")
        with open(path, "w") as f:
            f.write("dummy")
        from meshxcad.__main__ import load_mesh
        with pytest.raises(ValueError, match="Unsupported mesh format"):
            load_mesh(path)
