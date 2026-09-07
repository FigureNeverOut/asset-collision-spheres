"""CPU regressions for the opt-in policy; real assets/native checks live in examples."""

from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import trimesh

from asset_collision_spheres.algorithms import auto_geometry as auto
from asset_collision_spheres.algorithms.box_features import (
    area_length,
    detect_box_features,
)
from asset_collision_spheres.algorithms.joint_selection import VERSION, normalize
from asset_collision_spheres.adapters.fastsim import (
    fit_mesh_attachment,
    prepare_request,
)
from asset_collision_spheres.geometry.material import (
    _closest_point_exact,
    sphere_clearance,
)
from asset_collision_spheres.geometry.validation import check_spheres


def mesh(sections=24):
    return trimesh.creation.revolve(
        np.array(
            [[0, 0], [0.12, 0], [0.12, 0.2], [0.113, 0.2], [0.113, 0.007], [0, 0.007]]
        ),
        sections=sections,
    )


def config(**kw):
    return dict(
        sphere_budget=16,
        max_outward_offset_m=0.006,
        candidate_sample_count=500,
        patch_count=16,
        seed=5,
        shape_hint="auto",
        **kw,
    )


@pytest.fixture(autouse=True)
def small_diagnostics(monkeypatch):
    monkeypatch.setattr(
        auto,
        "evaluate_auto_geometry_spheres",
        partial(auto.evaluate_auto_geometry_spheres, sample_count=600),
    )


def test_legacy_dispatch_is_unchanged():
    raw = config()
    del raw["shape_hint"]
    result = auto.generate_auto_geometry_spheres(mesh(), raw)
    assert result.diagnostics["algorithm_version"] == auto.ALGORITHM_VERSION
    assert "effective_hard_limit" not in result.diagnostics


@pytest.mark.parametrize("hint", ["auto", "open_box"])
def test_material_supported_box_rotation_subdivision_and_tray(hint):
    original = mesh(4)
    moved = original.copy()
    moved.apply_transform(trimesh.transformations.euler_matrix(0.41, -0.72, 0.18))
    moved.apply_translation([0.3, -0.7, 0.2])
    tray = original.copy()
    tray.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1]))
    tray.apply_scale([1.8, 1, 0.25])
    for subject in (original, moved, original.subdivide(), tray):
        features = detect_box_features(subject, hint)
        assert features.enabled and features.confidence > 0.58
        assert len(features.anchors) >= 8
        assert len(features.segments) >= 16
        _, distances, _ = _closest_point_exact(subject, features.anchors)
        assert distances.max() < 1e-10
        for segment in features.segments:
            assert set(segment["support_vertex_ids"]) <= set(
                range(len(subject.vertices))
            )
    assert area_length(original) == pytest.approx(area_length(moved))
    assert area_length(original) == pytest.approx(area_length(original.subdivide()))


def test_generic_and_curved_fallback_do_not_invent_corners():
    assert not detect_box_features(mesh(4), "generic").enabled
    assert not detect_box_features(mesh(48), "auto").enabled
    result = auto.generate_auto_geometry_spheres(mesh(48), config())
    assert result.diagnostics["fallback_used"]
    assert result.diagnostics["feature_diagnostics"]["corner_anchor_count"] == 0


@pytest.mark.parametrize("seed", [0, 5, 42])
def test_fixed_multi_radius_mutual_exclusion_caps_cavity_and_seeds(seed):
    raw = config(system_policy={"outward_cost_weight": 0.006})
    raw["seed"] = seed
    before = deepcopy(raw)
    result = auto.generate_auto_geometry_spheres(mesh(), raw)
    info = result.diagnostics
    assert info["algorithm_version"] == VERSION and len(result.spheres) == 16
    assert raw == before
    assert len(set(info["selected_center_group_ids"])) == len(result.spheres)
    assert info["outward_expansion_max"] <= 0.006
    assert info["selected_large_radius_fraction"] < 1
    check_spheres(mesh(), result.spheres, 16, 0.006)
    assert sphere_clearance([[0, 0, 0.1]], result.spheres, 0.005)[0] > 0
    assert info["diagnostics"]["free_probe_intrusions_beyond_allowance"] == 0
    assert (
        auto.evaluate_auto_geometry_spheres(mesh(), result.spheres, raw)[
            "ordinary_sample_count"
        ]
        == 600
    )


def test_fixed_extra_never_breaks_budget_and_is_reported():
    raw = config(extra_face_spheres=8, max_spheres=32)
    result = auto.generate_auto_geometry_spheres(mesh(), raw)
    assert len(result.spheres) == 16
    assert result.diagnostics["extra_face_unfulfilled"] == 8


def test_adaptive_extra_appends_without_reseeding_and_respects_hard_max():
    raw = config(
        sphere_count_mode="adaptive",
        max_spheres=12,
        system_policy={"min_marginal_gain": 0.5},
    )
    a = auto.generate_auto_geometry_spheres(mesh(), raw)
    b = auto.generate_auto_geometry_spheres(mesh(), {**raw, "extra_face_spheres": 8})
    assert len(a.spheres) == 8 and a.diagnostics["count_stop_reason"] == "marginal_gain"
    np.testing.assert_array_equal(b.spheres[:8], a.spheres)
    assert len(b.spheres) == 12
    assert (
        b.diagnostics["extra_face_added"] == 4
        and b.diagnostics["extra_face_unfulfilled"] == 4
    )


def test_backend_capacity_policy_cache_and_generated_attachment():
    raw = config(sphere_count_mode="adaptive", max_spheres=100)
    del raw["sphere_budget"]
    a, digest_a = prepare_request(raw, capacity=12, seed=5)
    b, digest_b = prepare_request(raw, capacity=16, seed=5)
    c, digest_c = prepare_request(
        {**raw, "system_policy": {"outward_cost_power": 1}}, capacity=12, seed=5
    )
    assert len({digest_a, digest_b, digest_c}) == 3
    assert a["backend_sphere_capacity"] == 12 and b["backend_sphere_capacity"] == 16
    result = fit_mesh_attachment(mesh(), a)
    assert (
        len(result.spheres) <= 12
        and result.diagnostics["backend_sphere_capacity"] == 12
    )
    with pytest.raises(ValueError, match="actual backend"):
        prepare_request({**raw, "backend_sphere_capacity": 99}, capacity=12, seed=5)


def test_scale_aware_explicit_cap_zero_and_scale_covariance():
    subject = mesh()
    subject.apply_scale(0.4)  # first test below the absolute physical clamp
    raw = config(outward_offset_mode="scale_aware")
    raw["max_outward_offset_m"] = None
    a = normalize(subject, raw)
    bigger = subject.copy()
    bigger.apply_scale(1.5)
    b = normalize(bigger, raw)
    assert b["max_outward_offset_m"] == pytest.approx(a["max_outward_offset_m"] * 1.5)
    huge = subject.copy()
    huge.apply_scale(10)
    assert normalize(huge, raw)["max_outward_offset_m"] == 0.020
    limited = normalize(bigger, {**raw, "max_outward_offset_m": 0.002})
    assert limited["max_outward_offset_m"] == 0.002
    zero = auto.generate_auto_geometry_spheres(
        subject, {**raw, "max_outward_offset_m": 0}
    )
    assert zero.diagnostics["outward_expansion_max"] == 0
    assert zero.diagnostics["candidates_considered"] == zero.diagnostics["centre_count"]


def test_protected_corner_support_survives_bounded_repair():
    result = auto.generate_auto_geometry_spheres(mesh(4), config(repair_rounds=2))
    info = result.diagnostics
    unsupported = set(info["feature_diagnostics"]["unsupported_corner_ids"])
    assert not unsupported.intersection(info["protected_corner_ids"])
    assert len(info["protected_corner_ids"]) > 0 and len(info["repair_log"]) <= 2


def test_iteration_limit_returns_partial_with_review():
    result = auto.generate_auto_geometry_spheres(
        mesh(), config(system_policy={"max_iterations": 1})
    )
    assert len(result.spheres) == 1
    assert result.diagnostics["count_stop_reason"] == "iteration_limit"
    assert result.diagnostics["review_required"]


@pytest.mark.parametrize(
    "bad",
    [
        {"shape_hint": "mug"},
        {"sphere_count_mode": "unlimited"},
        {"max_spheres": True},
        {"max_spheres": 8},
        {"extra_face_spheres": -1},
        {"outward_offset_mode": "category"},
        {"system_policy": {"system_max_spheres": 999}},
        {"system_policy": {"outward_cost_weight": float("nan")}},
    ],
)
def test_invalid_new_config_fails(bad):
    with pytest.raises(ValueError):
        normalize(mesh(), {**config(), **bad})


def test_open_material_mesh_still_rejected():
    subject = mesh()
    subject.update_faces(np.arange(len(subject.faces) - 1))
    with pytest.raises(ValueError, match="sign is uncertain"):
        auto.generate_auto_geometry_spheres(subject, config())


def test_native_blocks_missing_important_structure_not_ordinary_gaps():
    raw, _ = prepare_request({**config(), "sphere_budget": 1}, capacity=16, seed=5)
    result = auto.generate_auto_geometry_spheres(mesh(4), raw)
    assert result.diagnostics["review_required"]
    with pytest.raises(ValueError, match="Important detected structure"):
        fit_mesh_attachment(mesh(4), raw)


def test_adaptive_cannot_stop_with_only_feature_seed_balls():
    raw = config(
        sphere_count_mode="adaptive",
        max_spheres=64,
        system_policy={"min_marginal_gain": 0.5},
    )
    result = auto.generate_auto_geometry_spheres(mesh(4), raw)
    assert result.diagnostics["feature_seed_count"] > 0
    assert result.diagnostics["feature_seed_count"] <= 0.35 * len(result.spheres)


def test_global_ablation_repair_does_not_reintroduce_patch_balance():
    raw = {**config(mode="global", repair_rounds=1), "shape_hint": "generic"}
    a = auto.generate_auto_geometry_spheres(mesh(), {**raw, "balance_strength": 0})
    b = auto.generate_auto_geometry_spheres(mesh(), {**raw, "balance_strength": 99})
    np.testing.assert_array_equal(a.spheres, b.spheres)
