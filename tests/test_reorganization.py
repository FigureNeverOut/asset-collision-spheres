"""Contracts affected by moving files: legacy identity, config paths and startup."""
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("old,new", [
    ("auto_geometry_spheres", "algorithms.auto_geometry"),
    ("mesh_region_spheres", "algorithms.mesh_regions"),
    ("attachment_spheres", "algorithms.analytic"),
    ("solid_box_regions", "algorithms.solid_box"),
    ("mesh_attachment", "adapters.fastsim"),
    ("usd", "loaders.usd"),
])
def test_legacy_module_is_the_same_module(old, new):
    assert import_module("asset_collision_spheres." + old) is import_module("asset_collision_spheres." + new)


def test_preview_import_keeps_kit_startup_free_of_standalone_libraries(tmp_path):
    code = """
import sys
import asset_collision_spheres.preview.mesh
forbidden = {'numpy', 'scipy', 'trimesh', 'pxr', 'torch', 'isaacsim', 'omni'}
assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_bundled_and_local_configs_are_distinct(monkeypatch, tmp_path):
    from asset_collision_spheres.paths import config_path
    name = "bowl/bowl_001_auto_geometry.json"
    bundled = config_path(name, local=False)
    assert not Path(json.loads(bundled.read_text())["source"]["usd_path"]).is_absolute()
    local = tmp_path / name
    local.parent.mkdir()
    local.write_text('{"test_override": true}')
    monkeypatch.setenv("ASSET_SPHERES_CONFIG_DIR", str(tmp_path))
    assert config_path(name) == local
    assert config_path(name, local=False) == bundled


def test_relative_usd_preset_builds_from_another_directory(tmp_path):
    pytest.importorskip("pxr")
    import trimesh
    from pxr import Usd, UsdGeom
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    stage = Usd.Stage.CreateNew(str(asset_dir / "box.usda"))
    shape = trimesh.creation.box([0.1, 0.08, 0.04])
    mesh = UsdGeom.Mesh.Define(stage, "/Box")
    mesh.CreatePointsAttr(shape.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr(shape.faces.ravel().tolist())
    stage.GetRootLayer().Save()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    preset = config_dir / "box.json"
    preset.write_text(json.dumps({
        "source": {"usd_path": "../assets/box.usda", "mesh_prim": "/Box", "unit_scale_m": 1, "up_axis": "Z"},
        "generation": {"sphere_budget": 8, "max_outward_offset_m": 0.004, "candidate_sample_count": 500, "patch_count": 8},
    }))
    subprocess.run([
        str(Path(sys.executable).parent / "asset-spheres-preview"),
        "--preset", str(preset), "--build-only", "--output", str(tmp_path / "result"),
    ], cwd=asset_dir, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "result/auto_geometry_8.json").read_text())
    assert len(report["spheres_preview_local_m"]) == 8
    assert (tmp_path / "result/auto_geometry_8.usda").is_file()
