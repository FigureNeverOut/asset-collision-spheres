import hashlib
import json
from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import trimesh

from asset_collision_spheres import auto_geometry_spheres as auto
from asset_collision_spheres.mesh_attachment import (
    characteristic_length,
    check_spheres,
    fit_mesh_attachment,
    mesh_fingerprint,
    prepare_request,
)
from asset_collision_spheres.mesh_region_spheres import sphere_clearance


def cup(sections=24):
    return trimesh.creation.revolve(
        np.array([[0, 0], [0.04, 0], [0.04, 0.1], [0.035, 0.1], [0.035, 0.008], [0, 0.008]]), sections=sections
    )


def config(**kwargs):
    return {
        "sphere_budget": 16,
        "max_outward_offset_m": 0.004,
        "candidate_sample_count": 500,
        "patch_count": 16,
        "seed": 5,
        **kwargs,
    }


@pytest.fixture(autouse=True)
def small_independent_diagnostics(monkeypatch):
    # Only unit-test diagnostics are shortened; real 1e artifact uses 12,000.
    monkeypatch.setattr(
        auto, "evaluate_auto_geometry_spheres", partial(auto.evaluate_auto_geometry_spheres, sample_count=600)
    )


def test_no_annotations_deterministic_balanced_and_global_share_candidates():
    mesh = cup()
    request = config(repair_rounds=1)
    before = deepcopy(request)
    a = auto.generate_auto_geometry_spheres(mesh, request)
    b = auto.generate_auto_geometry_spheres(mesh, request)
    global_result = auto.generate_auto_geometry_spheres(mesh, {**request, "mode": "global"})
    np.testing.assert_array_equal(a.spheres, b.spheres)
    assert a.diagnostics["candidate_pool_sha256"] == global_result.diagnostics["candidate_pool_sha256"]
    assert request == before
    assert len(a.spheres) == 16
    assert all(trace["local_balance_gain"] == 0 for trace in global_result.diagnostics["selection_trace"])
    assert any(trace["local_balance_gain"] > 0 for trace in a.diagnostics["selection_trace"])
    assert a.diagnostics["diagnostics"]["ordinary_contact_misses"] > 0
    assert a.diagnostics["validation_status"] == "finite_checks_passed"
    assert a.diagnostics["diagnostics"]["free_probe_intrusions_beyond_allowance"] == 0
    path = np.c_[np.zeros(40), np.zeros(40), np.linspace(0.025, 0.13, 40)]
    assert np.all(sphere_clearance(path, a.spheres, 0.005) > 0)
    assert len(a.diagnostics["repair_log"]) <= 1


def test_patches_never_join_disconnected_nearby_walls_through_air():
    first = trimesh.creation.box(extents=[0.1, 0.1, 0.002])
    second = first.copy()
    second.apply_translation([0, 0, 0.004])
    mesh = trimesh.util.concatenate([first, second])
    labels, area, components, _ = auto.local_patches(mesh, np.arange(len(mesh.faces)), 8)
    for label in np.unique(labels):
        assert len(set(components[labels == label])) == 1
    assert area.sum() == pytest.approx(mesh.area)


def test_rigid_transform_covariant_candidate_pool_and_scale():
    mesh = cup()
    request = auto.normalize_config(mesh, config())
    a = auto.candidate_pool(mesh, request)
    transform = trimesh.transformations.euler_matrix(0.43, -0.52, 0.16)
    transform[:3, 3] = [2, -3, 1]
    moved = mesh.copy()
    moved.apply_transform(transform)
    b = auto.candidate_pool(moved, auto.normalize_config(moved, config()))
    assert characteristic_length(moved) == pytest.approx(characteristic_length(mesh))
    np.testing.assert_allclose(b[0], trimesh.transform_points(a[0], transform), atol=1e-10)
    np.testing.assert_allclose(b[1], a[1], atol=1e-10)
    scaled = mesh.copy()
    scaled.apply_scale(2)
    c = auto.candidate_pool(scaled, auto.normalize_config(scaled, config(max_outward_offset_m=0.008)))
    np.testing.assert_allclose(c[0], a[0] * 2, atol=1e-10)
    np.testing.assert_allclose(c[1], a[1] * 2, atol=1e-10)


def test_planar_subdivision_does_not_change_scale_or_use_face_quotas():
    mesh = cup(12)
    fine = mesh.subdivide()
    assert characteristic_length(mesh) == pytest.approx(characteristic_length(fine))
    a = auto.generate_auto_geometry_spheres(mesh, config())
    b = auto.generate_auto_geometry_spheres(fine, config())
    assert len(a.spheres) == len(b.spheres) == 16
    assert (
        abs(
            a.diagnostics["diagnostics"]["contact_detection_fraction"]
            - b.diagnostics["diagnostics"]["contact_detection_fraction"]
        )
        < 0.15
    )
    assert sum(p["area_m2"] for p in a.diagnostics["patches"]) == pytest.approx(
        sum(p["area_m2"] for p in b.diagnostics["patches"])
    )


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_different_seeds_still_preserve_material_and_open_axis(seed):
    mesh = cup()
    result = auto.generate_auto_geometry_spheres(mesh, config(seed=seed))
    check_spheres(mesh, result.spheres, 16, 0.004)
    assert sphere_clearance([[0, 0, 0.06]], result.spheres, 0.005)[0] > 0


def test_whole_component_omission_is_visible_not_concealed_as_success():
    mesh = trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=[0.03] * 3),
            trimesh.creation.box(extents=[0.03] * 3, transform=trimesh.transformations.translation_matrix([0.1, 0, 0])),
        ]
    )
    result = auto.generate_auto_geometry_spheres(mesh, config(sphere_budget=1))
    assert len(result.diagnostics["missing_material_components"]) == 1
    assert result.diagnostics["zero_contact_patch_ids"]
    assert result.diagnostics["review_required"]
    request, _ = prepare_request(config(sphere_budget=1), capacity=16, seed=5)
    with pytest.raises(ValueError, match="Entire material components"):
        fit_mesh_attachment(mesh, request)


@pytest.mark.parametrize("shape", ["bowl", "open_box", "pipe"])
def test_other_hollow_meshes_without_semantic_configuration(shape):
    if shape == "pipe":
        mesh = trimesh.creation.annulus(r_min=0.03, r_max=0.035, height=0.1, sections=24)
    elif shape == "bowl":
        mesh = trimesh.creation.revolve(
            np.array([[0, 0], [0.02, 0], [0.05, 0.08], [0.045, 0.08], [0.016, 0.006], [0, 0.006]]), sections=24
        )
    else:
        mesh = cup(4)
    result = auto.generate_auto_geometry_spheres(mesh, config())
    check_spheres(mesh, result.spheres, 16, 0.004)
    assert not result.diagnostics["missing_material_components"]
    assert result.diagnostics["diagnostics"]["free_probe_intrusions_beyond_allowance"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"regions": []},
        {"axis": "z"},
        {"sphere_budget": True},
        {"max_outward_offset_m": float("nan")},
        {"repair_rounds": 999},
    ],
)
def test_bad_or_manual_config_rejected(bad):
    with pytest.raises((ValueError, KeyError)):
        auto.generate_auto_geometry_spheres(cup(), config(**bad))


def test_expansion_has_no_universal_default_and_open_mesh_rejected():
    with pytest.raises(KeyError, match="max_outward"):
        auto.normalize_config(cup(), {"sphere_budget": 16})
    mesh = cup()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="sign is uncertain"):
        auto.generate_auto_geometry_spheres(mesh, config())


@pytest.mark.parametrize("sphere", [[[0, 0, 0.05, 0.08]], [[0, 0, 0, 0]], [[0, 0, 0, float("inf")]], []])
def test_invalid_or_cavity_filling_spheres_fail_closed(sphere):
    with pytest.raises(ValueError):
        check_spheres(cup(), sphere, 16, 0.004)


def test_precomputed_hash_mesh_identity_and_capacity_checked(tmp_path):
    mesh = trimesh.creation.box(extents=[0.1] * 3)
    artifact = {
        "schema": "fastsim/mesh-spheres/1",
        "mesh_sha256": mesh_fingerprint(mesh),
        "spheres_object_local_m": [[0, 0, 0, 0.04]],
        "max_outward_offset_m": 0.004,
    }
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(artifact))
    raw = {
        "mode": "precomputed",
        "artifact_path": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    request, _ = prepare_request(raw, capacity=16, seed=5)
    result = fit_mesh_attachment(mesh, request)
    assert len(result.spheres) == 1  # unused native slots must remain negative
    other = mesh.copy()
    other.apply_scale([1, 1.2, 1])
    with pytest.raises(ValueError, match="do not match"):
        fit_mesh_attachment(other, request)
    path.write_text(json.dumps({**artifact, "spheres_object_local_m": []}))
    with pytest.raises(ValueError, match="SHA256"):
        prepare_request(raw, capacity=16, seed=5)
    with pytest.raises(ValueError, match="capacity"):
        prepare_request({**raw, "sphere_budget": 64}, capacity=16, seed=5)
