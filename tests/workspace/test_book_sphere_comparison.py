"""Actual-asset preview alignment and static pose regressions (no GPU needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("trimesh")


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
    module = import_module("asset_collision_spheres.adapters.workspace.book_comparison")
    if not (module.PROJECT / "run.yaml").is_file():
        pytest.skip("Local historical 1g execution is unavailable")
    return module


def test_actual_book_visual_only_repair_preserves_source_and_collision(
    preview, tmp_path
):
    from pxr import Usd, UsdPhysics

    source = preview.source_state()["book_asset"]
    original_bytes = source.read_bytes()
    original = Usd.Stage.Open(str(source))
    path, proof = preview.aligned_book_asset(source, tmp_path)
    repaired = Usd.Stage.Open(str(path))
    assert repaired.GetRootLayer().defaultPrim == original.GetDefaultPrim().GetName()
    assert proof["new_bounds_error_m"] <= 2e-6
    assert proof["old_center_offset_m"][2] == pytest.approx(0.082018994, abs=1e-8)
    for prim in original.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            other = repaired.GetPrimAtPath(prim.GetPath())
            for attribute in prim.GetAttributes():
                if attribute.GetName().startswith(("physics:", "physx")):
                    assert (
                        other.GetAttribute(attribute.GetName()).Get() == attribute.Get()
                    )
    np.testing.assert_array_equal(
        preview.mesh_data(original, True).vertices,
        preview.mesh_data(repaired, True).vertices,
    )
    np.testing.assert_array_equal(
        preview.mesh_data(original, True).faces,
        preview.mesh_data(repaired, True).faces,
    )
    assert source.read_bytes() == original_bytes
    # Rebuilding our generated overlay must also be safe and idempotent.
    assert preview.aligned_book_asset(source, tmp_path)[1]["new_bounds_error_m"] <= 2e-6


def test_static_scene_transforms_spheres_once_and_retains_purple_initial_mesh(
    preview, tmp_path, monkeypatch
):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    run_bytes = (preview.PROJECT / "run.yaml").read_bytes()
    balls = [[0.03, 0.001, 0.08, 0.006], [-0.05, 0, 0.05, 0.01]]
    monkeypatch.setattr(
        preview,
        "fit_sets",
        lambda *args: (
            {"baseline": balls},
            {"independent_mesh_holdouts": {}, "native_attachment_executed": False},
        ),
    )
    args = SimpleNamespace(output=tmp_path, method="baseline", budget=64)
    path = preview.build(args)
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    book_prim = stage.GetPrimAtPath("/World/BookAtAttach")
    assert sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(book_prim)) >= 2
    book_world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath("/World/BookAtAttach")
    )
    for index, (x, y, z, r) in enumerate(balls):
        prim = stage.GetPrimAtPath(f"/World/CollisionSpheres/sphere_{index:03d}")
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(prim).ExtractTranslation(),
            book_world.Transform(Gf.Vec3d(x, y, z)),
            atol=1e-9,
        )
        assert UsdGeom.Sphere(prim).GetRadiusAttr().Get() == pytest.approx(r)
    initial = stage.GetPrimAtPath("/World/BookInitialPosition")
    expected = preview.source_state()["run"]["scenario"]["scene"]["objects"]["book_a"][
        "pose"
    ]
    np.testing.assert_allclose(
        cache.GetLocalToWorldTransform(initial), preview.matrix(expected), atol=1e-9
    )
    assert UsdGeom.Imageable(initial).ComputeVisibility() == UsdGeom.Tokens.inherited
    for prim in stage.Traverse():
        assert not prim.IsA(UsdPhysics.Joint)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            assert not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert not UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
    evidence = json.loads(path.with_suffix(".json").read_text())
    assert not evidence["native_attachment_executed"]
    assert (preview.PROJECT / "run.yaml").read_bytes() == run_bytes
