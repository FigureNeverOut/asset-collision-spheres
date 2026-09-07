"""Matched inputs, unchanged scene, sphere budget and shared coordinate frames."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom


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
    module = import_module("asset_collision_spheres.adapters.workspace.curobo_matched")
    monkeypatch.setattr(module, "OUTPUT", module.LEGACY_OUTPUT)
    return module


def stripped_stage(path, sphere_root):
    original = Usd.Stage.Open(str(path))
    copy = Usd.Stage.CreateInMemory()
    copy.GetRootLayer().ImportFromString(original.GetRootLayer().ExportToString())
    copy.RemovePrim(sphere_root)
    copy.RemovePrim("/World/MatchedSphereMaterial")
    return copy.GetRootLayer().ExportToString()


@pytest.mark.parametrize("asset", ["mug", "bowl"])
def test_saved_pair_has_equal_inputs_and_unchanged_non_sphere_scene(preview, asset):
    report_path = preview.OUTPUT / asset / "comparison.json"
    if not report_path.exists():
        pytest.skip("Build both matched comparisons first")
    source, _, improved, config, fingerprint = preview.inputs(asset)
    report = json.loads(report_path.read_text())
    assert report["sphere_budget"] == 32
    assert report["seed"] == config["seed"] == 5
    assert report["mesh_sha256"] == fingerprint
    assert not report["native_attachment_or_physics_tested_this_run"]
    assert report["baseline_fit"]["debug_info"]["requested_n_spheres"] == 32
    assert not report["baseline_fit"]["debug_info"]["containment_enforced"]
    np.testing.assert_array_equal(
        report["methods"]["auto_geometry"]["spheres_object_local_m"], improved
    )
    sphere_root = "/World/CollisionSpheres" if asset == "mug" else "/World/Spheres"
    expected_scene = stripped_stage(source, sphere_root)
    for method, evidence in report["methods"].items():
        assert 1 <= evidence["actual_count"] <= 32
        path = preview.verify(asset, method)
        assert stripped_stage(path, sphere_root) == expected_scene
        stage = Usd.Stage.Open(str(path))
        cache = UsdGeom.XformCache()
        frame = (
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/MugAtAttach"))
            if asset == "mug"
            else Gf.Matrix4d(1)
        )
        for i, sphere in enumerate(evidence["spheres_object_local_m"]):
            prim = stage.GetPrimAtPath(f"{sphere_root}/sphere_{i:03d}")
            np.testing.assert_allclose(
                cache.GetLocalToWorldTransform(prim).ExtractTranslation(),
                frame.Transform(Gf.Vec3d(*sphere[:3])),
                atol=1e-12,
            )
            assert UsdGeom.Sphere(prim).GetRadiusAttr().Get() == sphere[3]
    a, b = (v["evaluation"] for v in report["methods"].values())
    assert a["contact_probe_count"] == b["contact_probe_count"]
    assert (
        a["free_probe_count_beyond_new_allowance"]
        == b["free_probe_count_beyond_new_allowance"]
    )
