"""Compare artifacts without changing the approved 1e output or runtime physics."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
yaml = pytest.importorskip("yaml")


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastsim_plugin_mission") is None,
    reason="Requires the existing FastSim workspace environment",
)


@pytest.fixture
def module(monkeypatch):
    from importlib import import_module
    from asset_collision_spheres.paths import workspace_root
    if workspace_root(required=False) is None:
        pytest.skip("FastSim workspace is unavailable")
    module = import_module("asset_collision_spheres.adapters.workspace.mug_auto")
    return module


def test_32_config_changes_only_budget(module):
    directory = module.config_path("mug/1e_mug_auto_geometry.json").parent
    original = json.loads((directory / "1e_mug_auto_geometry.json").read_text())
    reduced = json.loads((directory / "1e_mug_auto_geometry_32.json").read_text())
    assert reduced == {**original, "sphere_budget": 32}


def test_saved_32_spheres_use_same_candidates_and_native_capacity(module):
    output = module.LEGACY_OUTPUT.parent / "mug_auto_geometry_32"
    if not (output / "auto_geometry_native.json").is_file():
        pytest.skip("Build the 32-sphere native preview first")
    reference = json.loads((module.LEGACY_OUTPUT / "auto_geometry.json").read_text())
    reduced = json.loads((output / "auto_geometry.json").read_text())
    native = json.loads((output / "auto_geometry_native.json").read_text())
    assert len(reduced["spheres_object_local_m"]) == 32
    assert (
        native["actual_count"]
        == native["requested_budget"]
        == native["configured_capacity"]
        == 32
    )
    assert (
        reduced["fit"]["candidate_pool_sha256"]
        == reference["fit"]["candidate_pool_sha256"]
    )
    assert (
        native["sentinel"]["passed"]
        and native["sentinel"]["detached_slots_all_negative"]
    )
    assert native["reattach_cache_hit"]
    assert native["preview_vs_native_center_max_m"] < 2e-5


def test_approved_files_have_not_changed(module):
    assert sha256(module.FROZEN.read_bytes()).hexdigest() == module.FROZEN_SHA
    assert (
        sha256(module.preview.DEFAULT_STRUCTURE_CONFIG.read_bytes()).hexdigest()
        == "b39dd5175b651b912a3641583e5ae4c4bfd4c0662ed584b31be83e7f97a2a1bd"
    )
    assert (
        sha256(module.FROZEN.with_suffix(".usda").read_bytes()).hexdigest()
        == "0ac20750253c4ac4ca5cfc36004cdd95e81d5a75819edde59b4e8deb74e7a7ed"
    )


def test_experiment_run_changes_only_sphere_config_and_resolves_offline(
    module, tmp_path
):
    source = yaml.safe_load(module.preview.RUN.read_text())
    original = module.preview.RUN.read_bytes()
    args = SimpleNamespace(
        method="auto_geometry",
        output=tmp_path,
        diagnostic_budget=64,
        geometry_config=module.config_path("mug/1e_mug_auto_geometry.json"),
    )
    path = module.write_runtime_config(args)
    changed = yaml.safe_load(path.read_text())
    expected = deepcopy(source)
    old = module.preview.find_solver_config(expected)
    old.update(
        attached_object_sphere_budget=64,
        attached_object_sphere_method="mesh",
        attached_object_sphere_fallback="error",
        attached_object_mesh_configs={
            "mug": json.loads(args.geometry_config.read_text())
        },
    )
    assert changed == expected
    assert module.preview.RUN.read_bytes() == original
    resolved = module.materialize_offline_run(path)
    runtime = yaml.safe_load(resolved.read_text())
    assert runtime["scenario"] == source["scenario"]
    assert Path(module.preview.find_solver_config(runtime)["robot_config"]).is_file()


def test_saved_three_way_comparison_native_spheres_and_free_paths(module):
    from pxr import Gf, Usd, UsdGeom

    modes = ["manual_frozen", "global", "auto_geometry"]
    if not all((module.LEGACY_OUTPUT / (mode + "_native.json")).exists() for mode in modes):
        pytest.skip("Run three real native 1e previews first")
    reports = {}
    for mode in modes:
        evidence = json.loads((module.LEGACY_OUTPUT / (mode + ".json")).read_text())
        fit = evidence["fit"]
        reports[mode] = fit
        native = json.loads((module.LEGACY_OUTPUT / (mode + "_native.json")).read_text())
        assert native["actual_count"] == native["configured_capacity"] == 64
        assert (
            native["sentinel"]["passed"]
            and native["sentinel"]["detached_slots_all_negative"]
        )
        assert native["reattach_cache_hit"]
        assert native["preview_vs_native_center_max_m"] < 2e-5
        assert all(
            path["sphere_false_positive_count"] == 0
            for path in fit["heldout_manual_labels_only"]["free_probe_paths"].values()
        )
        stage = Usd.Stage.Open(str(module.LEGACY_OUTPUT / (mode + ".usda")))
        transforms = UsdGeom.XformCache()
        for i, sphere in enumerate(native["world_spheres"]):
            prim = stage.GetPrimAtPath(f"/World/CollisionSpheres/sphere_{i:03d}")
            center = transforms.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(0))
            np.testing.assert_allclose(
                center, sphere["center_world_m"], atol=1e-8, rtol=0
            )
        assert stage.GetPrimAtPath("/World/MugInitialPosition/Outline")
    assert (
        reports["global"]["candidate_pool_sha256"]
        == reports["auto_geometry"]["candidate_pool_sha256"]
    )
    frozen = json.loads(module.FROZEN.read_text())
    manual = json.loads((module.LEGACY_OUTPUT / "manual_frozen.json").read_text())
    assert manual["spheres_object_local_m"] == frozen["spheres_object_local_m"]
