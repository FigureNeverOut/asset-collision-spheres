import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import trimesh

from asset_collision_spheres import load_mesh


def test_installed_cli_generates_portable_report_and_refuses_overwrite(tmp_path):
    mesh = trimesh.creation.box(extents=[0.1, 0.08, 0.04])
    mesh.export(tmp_path / "box.obj")
    config = {
        "sphere_budget": 8, "max_outward_offset_m": 0.004,
        "candidate_sample_count": 500, "patch_count": 8, "seed": 5,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    command = [
        str(Path(sys.executable).parent / "asset-spheres"),
        "--mesh", "box.obj", "--unit-scale-m", "1",
        "--config", "config.json", "--output", "result.json",
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["schema"] == "asset-collision-spheres/1"
    assert result["coordinate_frame"] == "input_mesh_axes_m"
    assert np.asarray(result["spheres_m"]).shape == (8, 4)
    assert result["diagnostics"]["actual_count"] == 8
    assert not result["diagnostics"]["missing_material_components"]
    saved = (tmp_path / "result.json").read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert (tmp_path / "result.json").read_bytes() == saved


@pytest.mark.parametrize("extension", ["obj", "ply", "stl"])
def test_file_loader_preserves_physical_size_and_material(extension, tmp_path):
    mesh = trimesh.creation.box(extents=[100, 80, 40])
    path = tmp_path / f"box.{extension}"
    mesh.export(path)
    loaded = load_mesh(path, unit_scale_m=0.001)
    assert loaded.is_watertight
    np.testing.assert_allclose(loaded.extents, [0.1, 0.08, 0.04])
    assert loaded.volume == pytest.approx(0.1 * 0.08 * 0.04)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True])
def test_loader_rejects_invalid_units_before_loading(scale):
    with pytest.raises(ValueError, match="unit_scale"):
        load_mesh("unused.obj", unit_scale_m=scale)


def test_core_import_does_not_load_simulation_or_usd_packages(tmp_path):
    code = (
        "import sys; import asset_collision_spheres; "
        "forbidden = ('fastsim', 'curobo', 'isaacsim', 'omni', 'pxr', 'torch'); "
        "assert not any(n.split('.')[0] in forbidden for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_usd_loader_applies_authored_transform_units_and_axis(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom
    from asset_collision_spheres.usd import load_material_mesh

    mesh = trimesh.creation.box(extents=[100, 80, 40])
    path = tmp_path / "box.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # Explicit scale takes precedence.
    prim = UsdGeom.Mesh.Define(stage, "/Box")
    prim.CreatePointsAttr(mesh.vertices.tolist())
    prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    prim.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    stage.GetRootLayer().Save()
    source = {
        "usd_path": str(path), "mesh_prim": "/Box", "unit_scale_m": 0.001,
        "expected_extents_m": [0.1, 0.08, 0.04],
    }
    loaded, metadata = load_material_mesh({"source": source})
    np.testing.assert_allclose(loaded.centroid, [0.01, -0.03, 0.02], atol=1e-8)
    np.testing.assert_allclose(loaded.extents, [0.1, 0.04, 0.08])
    assert metadata["authored_meters_per_unit"] == pytest.approx(0.01)
    source["expected_extents_m"] = [1, 1, 1]
    with pytest.raises(ValueError, match="extents"):
        load_material_mesh({"source": source})
