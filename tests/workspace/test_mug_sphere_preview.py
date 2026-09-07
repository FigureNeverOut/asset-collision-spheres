"""Static viewer checks. Real-asset integration skips outside this workspace."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("scipy")
trimesh = pytest.importorskip("trimesh")


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastsim_plugin_mission") is None,
    reason="Requires the existing FastSim workspace environment",
)


@pytest.fixture
def preview(monkeypatch):
    from importlib import import_module
    from asset_collision_spheres.paths import workspace_root
    if workspace_root(required=False) is None:
        pytest.skip("FastSim workspace is unavailable")
    module = import_module("asset_collision_spheres.adapters.workspace.mug")
    return module


def test_alignment_rejects_shift_instead_of_recentering_spheres(preview):
    visual = trimesh.creation.box(extents=(0.1, 0.13, 0.115))
    collision = visual.copy()
    assert (
        preview.assert_aligned(collision, visual)["visual_collision_vertex_error_m"]
        == 0
    )
    collision.apply_translation([0, 0, 0.082])
    with pytest.raises(RuntimeError, match="do not align"):
        preview.assert_aligned(collision, visual)


def test_mission_quaternion_order(preview):
    pose = preview.inline_pose({"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]})
    assert pose == {"xyz_m": [1, 2, 3], "quat_xyzw": [0, 0, 0, 1]}


def test_real_1e_static_attachment_uses_one_transform(tmp_path, monkeypatch, preview):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not preview.RUN.is_file() or not (preview.CATALOG / "mug-000.yaml").is_file():
        pytest.skip("Local 1e runtime assets unavailable")
    state = preview.load_state()
    original_run = preview.RUN.read_bytes()
    # Geometry/pose regression independent of expensive and stochastic fitting.
    balls = [[0.01, 0.02, 0.03, 0.005], [-0.015, -0.01, 0.09, 0.008]]
    monkeypatch.setattr(
        preview, "fit_baseline", lambda *args: (balls, {"actual_count": 2})
    )
    args = SimpleNamespace(
        method="baseline",
        diagnostic_budget=64,
        device="cpu",
        output=tmp_path,
        hide_visual=False,
    )
    path = preview.build(args)
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    mug_world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/MugAtAttach")
    )
    anchor = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/Robot/right_gripper_center")
    )
    np.testing.assert_allclose(
        np.array(preview.matrix(state["grasp_pose"]) * mug_world),
        np.array(anchor),
        atol=1e-9,
    )
    for index, (x, y, z, radius) in enumerate(balls):
        sphere = stage.GetPrimAtPath(f"/World/CollisionSpheres/sphere_{index:03d}")
        actual = cache.GetLocalToWorldTransform(sphere).ExtractTranslation()
        expected = mug_world.Transform(Gf.Vec3d(x, y, z))
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        assert UsdGeom.Sphere(sphere).GetRadiusAttr().Get() == pytest.approx(radius)
    np.testing.assert_allclose(
        mug_world.ExtractTranslation(),
        np.array(state["initial_pose"]["xyz_m"]) + [0, 0, 0.1],
        atol=0.001,
    )
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            assert not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert not prim.IsA(UsdPhysics.Joint)  # joint display copies are inactive
    evidence = json.loads((tmp_path / "baseline.json").read_text())
    assert not evidence["analytic_cup_specialization_used"]
    assert not evidence["physics_and_planning_executed"]
    assert evidence["alignment"]["visual_collision_vertex_error_m"] == 0
    assert preview.RUN.read_bytes() == original_run
    preview.load_state()  # catalog SHA256 checks confirm source assets not overwritten


def test_real_1e_mesh_regions_preserve_handle_and_report_holdout(preview):
    from pxr import Usd

    if not preview.RUN.is_file() or not (preview.CATALOG / "mug-000.yaml").is_file():
        pytest.skip("Local 1e runtime assets unavailable")
    state = preview.load_state()
    mesh = preview.mesh_data(Usd.Stage.Open(str(state["assets"]["mug"])), True)
    module = preview.load_mesh_region_generator()
    request = json.loads(preview.DEFAULT_STRUCTURE_CONFIG.read_text())
    result = module.generate_mesh_region_spheres(mesh, request)
    assert len(result.spheres) == 64
    assert {name: result.regions.count(name) for name in set(result.regions)} == {
        "wall": 34,
        "rim": 12,
        "bottom": 8,
        "handle": 10,
    }
    # Regression for parity-ray misclassification deep inside this scan cavity.
    probes = np.array(
        [[0, -0.017, 0.025], [0, -0.017, 0.04], [0, -0.017, 0.06], [0, 0.043, 0.057]]
    )
    assert np.all(module.material_depth(mesh, probes) < 0)
    validation = module.evaluate_mesh_spheres(mesh, result.spheres, request)
    for path in validation["free_probe_paths"].values():
        assert path["reference_collision_count"] == 0
        assert path["sphere_false_positive_count"] == 0
    assert validation["max_material_expansion_bound_m"] <= 0.006 + 1e-7
    if validation["collision_miss_count"]:
        assert validation["status"] == "holdout_failed"
    # The validator must continue testing omitted material, independent of what
    # the generator allocated. Removing the handle cannot make its tests vanish.
    no_handle = result.spheres[np.array(result.regions) != "handle"]
    ablated = module.evaluate_mesh_spheres(mesh, no_handle, request)
    assert (
        ablated["by_region"]["handle"]["reference_collision_count"]
        == validation["by_region"]["handle"]["reference_collision_count"]
    )
    assert (
        ablated["by_region"]["handle"]["miss_count"]
        > validation["by_region"]["handle"]["miss_count"]
    )
