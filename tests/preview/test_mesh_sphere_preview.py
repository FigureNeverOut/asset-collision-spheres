"""Standalone asset loader, shared display frame and isolated budget checks."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import trimesh
pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom


@pytest.fixture
def preview(monkeypatch):
    from importlib import import_module
    module = import_module("asset_collision_spheres.preview.mesh")
    return module


def box_source(tmp_path, *, reflected=False, open_mesh=False):
    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    root.AddScaleOp().Set(Gf.Vec3f(-1 if reflected else 1, 1, 1))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, "Y")
    box = trimesh.creation.box(extents=[20, 40, 60])
    faces = box.faces[:-1] if open_mesh else box.faces
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.CreatePointsAttr(box.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    stage.GetRootLayer().Save()
    return {
        "source": {
            "usd_path": str(path),
            "mesh_prim": "/Asset/Mesh",
            "unit_scale_m": 0.001,
        }
    }


@pytest.mark.parametrize("reflected", [False, True])
def test_explicit_units_root_transform_and_winding(preview, tmp_path, reflected):
    preset = box_source(tmp_path, reflected=reflected)
    source = Path(preset["source"]["usd_path"])
    before = source.read_bytes()
    mesh, provenance = preview.load_material_mesh(preset)
    np.testing.assert_allclose(mesh.extents, [0.02, 0.06, 0.04])
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), [0.01, -0.03, 0.02])
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert provenance["authored_meters_per_unit"] == 0.01
    assert provenance["effective_unit_scale_m"] == 0.001
    assert source.read_bytes() == before


def test_bad_size_and_open_material_fail_without_hull(preview, tmp_path):
    preset = box_source(tmp_path, open_mesh=True)
    with pytest.raises(ValueError, match="closed"):
        preview.load_material_mesh(preset)
    preset["source"]["expected_extents_m"] = [20, 40, 60]
    with pytest.raises(ValueError, match="physical size"):
        preview.load_material_mesh(preset)


def test_annotation_not_used_and_budget_only_override(preview):
    original, a = preview.read_preset(preview.DEFAULT_PRESET)
    _, b = preview.read_preset(preview.DEFAULT_PRESET, 64)
    assert a["sphere_budget"] == 32
    assert b == {**a, "sphere_budget": 64}
    assert not {"regions", "label_info", "object_category", "up_axis"}.intersection(a)
    source = Path(original["source"]["usd_path"])
    if not source.exists():
        pytest.skip("Local bowl fixture is unavailable")
    modified = deepcopy(original)
    modified["annotation_provenance_only"] = "/nonexistent/ignored.json"
    modified["label_info"] = {"invented": "not an algorithm input"}
    m1, _ = preview.load_material_mesh(original)
    m2, _ = preview.load_material_mesh(modified)
    np.testing.assert_array_equal(m1.vertices, m2.vertices)
    np.testing.assert_array_equal(m1.faces, m2.faces)


def test_usd_spheres_and_reference_share_frame(preview, tmp_path):
    mesh = trimesh.creation.box([0.1, 0.2, 0.3])
    spheres = np.array([[0.01, 0.02, 0.03, 0.006], [-0.02, 0, 0.01, 0.008]])
    path = tmp_path / "preview.usda"
    preview.write_stage(path, mesh, spheres)
    stage = Usd.Stage.Open(str(path))
    assert stage.GetPrimAtPath("/World/Spheres").IsA(UsdGeom.Xform)
    actual = np.asarray(
        UsdGeom.Mesh(stage.GetPrimAtPath("/World/ReferenceMesh")).GetPointsAttr().Get()
    )
    np.testing.assert_allclose(actual, mesh.vertices)
    cache = UsdGeom.XformCache()
    for i, sphere in enumerate(spheres):
        prim = stage.GetPrimAtPath(f"/World/Spheres/sphere_{i:03d}")
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(prim).ExtractTranslation(), sphere[:3]
        )
        assert UsdGeom.Sphere(prim).GetRadiusAttr().Get() == sphere[3]
    assert not any("Physics" in str(p.GetAppliedSchemas()) for p in stage.Traverse())


def test_saved_bowl_budgets_and_stale_guard(preview):
    preset, config = preview.read_preset(preview.DEFAULT_PRESET)
    output = preview.ROOT / "task2sim/runs/collision_ball_test/bowl_001_auto_geometry"
    reports = []
    for budget in (32, 64):
        path = output / f"auto_geometry_{budget}.usda"
        if not path.exists():
            pytest.skip("Generate bowl previews first")
        cfg = {**config, "sphere_budget": budget}
        preview.verify_existing(path, preset, cfg)
        report = json.loads(path.with_suffix(".json").read_text())
        assert len(report["spheres_preview_local_m"]) == budget
        assert not report["robot_attachment_tested"]
        assert not report["source"]["geometry_repair_applied"]
        reports.append(report)
    assert (
        reports[0]["fit"]["candidate_pool_sha256"]
        == reports[1]["fit"]["candidate_pool_sha256"]
    )
    with pytest.raises(ValueError, match="configuration differs"):
        preview.verify_existing(
            output / "auto_geometry_32.usda", preset, {**config, "sphere_budget": 64}
        )


def test_box_uses_same_generation_config_and_has_open_cavity(preview):
    preset_path = preview.config_path("box/box_167_auto_geometry.json")
    preset, config = preview.read_preset(preset_path)
    _, bowl_config = preview.read_preset(preview.DEFAULT_PRESET, 32)
    assert config == bowl_config
    source = Path(preset["source"]["usd_path"])
    if not source.exists():
        pytest.skip("Local box fixture unavailable")
    before = source.read_bytes()
    mesh, info = preview.load_material_mesh(preset)
    assert info["watertight"] and info["winding_consistent"]
    assert not info["geometry_repair_applied"]
    assert not info["annotation_used_for_generation"]
    center, extent = mesh.bounds.mean(axis=0), mesh.extents
    # Held-out checks, never generation inputs: five interior rays reach the bottom.
    for dx, dy in [(0, 0), (-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]:
        origin = center + [dx * extent[0], dy * extent[1], extent[2]]
        hits, _, _ = mesh.ray.intersects_location([origin], [[0, 0, -1]])
        assert len(hits) > 0
        assert (hits[:, 2].max() - mesh.bounds[0, 2]) / extent[2] < 0.1
    assert source.read_bytes() == before
    reports = []
    output = preview.ROOT / "task2sim/runs/collision_ball_test/box_167_auto_geometry"
    for budget in [32, 64]:
        path = output / f"auto_geometry_{budget}.usda"
        if not path.exists():
            pytest.skip("Generate box previews first")
        preview.verify_existing(path, preset, {**config, "sphere_budget": budget})
        report = json.loads(path.with_suffix(".json").read_text())
        assert len(report["spheres_preview_local_m"]) == budget
        reports.append(report)
    assert (
        reports[0]["fit"]["candidate_pool_sha256"]
        == reports[1]["fit"]["candidate_pool_sha256"]
    )
