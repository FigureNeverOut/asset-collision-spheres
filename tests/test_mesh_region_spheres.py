from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import trimesh

from asset_collision_spheres.mesh_region_spheres import (
    evaluate_mesh_spheres,
    generate_mesh_region_spheres,
    material_depth,
    validate_request,
)


def cup():
    return trimesh.creation.revolve(
        np.array([[0, 0], [0.04, 0], [0.04, 0.1], [0.035, 0.1], [0.035, 0.008], [0, 0.008]]), sections=48
    )


def config(mesh):
    return {
        "sphere_budget": 24,
        "max_outward_offset_m": 0.004,
        "expected_bounds_m": mesh.bounds.tolist(),
        "candidate_sample_count": 1500,
        "seed": 5,
        "probe_radius_m": 0.006,
        "probe_penetration_m": 0.0015,
        "regions": [
            {
                "name": "rim",
                "bounds_m": [[-0.041, -0.041, 0.09], [0.041, 0.041, 0.101]],
                "sphere_count": 8,
                "max_radius_m": 0.007,
            },
            {
                "name": "bottom",
                "bounds_m": [[-0.041, -0.041, -0.001], [0.041, 0.041, 0.009]],
                "sphere_count": 4,
                "max_radius_m": 0.009,
            },
            {
                "name": "wall",
                "bounds_m": [[-0.041, -0.041, -0.001], [0.041, 0.041, 0.101]],
                "sphere_count": 12,
                "max_radius_m": 0.007,
            },
        ],
        "free_probe_paths": [
            {"name": "opening", "start_m": [0, 0, 0.03], "end_m": [0, 0, 0.12], "radius_m": 0.005, "count": 20}
        ],
    }


def test_material_sign_matches_box_faces_edges_vertices_and_rigid_transform():
    mesh = trimesh.creation.box(extents=(0.08, 0.06, 0.1))
    points = np.r_[
        np.random.default_rng(52).uniform(-0.12, 0.12, (500, 3)), [[0, 0, 0], [0.04, 0.03, 0.05], [0.05, 0.04, 0.06]]
    ]
    q = np.abs(points) - mesh.extents / 2
    analytic = -(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0))
    np.testing.assert_allclose(material_depth(mesh, points), analytic, atol=1e-8)
    transform = trimesh.transformations.euler_matrix(0.6, -0.4, 0.3)
    transform[:3, 3] = [1, 2, -3]
    mesh.apply_transform(transform)
    moved = trimesh.transform_points(points, transform)
    np.testing.assert_allclose(material_depth(mesh, moved), analytic, atol=1e-8)


def test_open_cup_axis_is_free_and_wall_is_material():
    mesh = cup()
    assert mesh.is_watertight and mesh.is_winding_consistent
    points = np.array([[0, 0, 0.03], [0, 0, 0.07], [0, 0, 0.11], [0.037, 0, 0.05], [0, 0, 0.004]])
    depth = material_depth(mesh, points)
    assert np.all(depth[:3] < 0)
    assert np.all(depth[3:] > 0)


def test_fixed_counts_determinism_and_material_expansion_limit():
    mesh = cup()
    request = config(mesh)
    a = generate_mesh_region_spheres(mesh, request)
    b = generate_mesh_region_spheres(mesh, request)
    np.testing.assert_array_equal(a.spheres, b.spheres)
    assert len(a.spheres) == 24
    assert {name: a.regions.count(name) for name in set(a.regions)} == {"rim": 8, "bottom": 4, "wall": 12}
    depth = material_depth(mesh, a.spheres[:, :3])
    assert np.all(depth > 0)
    assert np.all(a.spheres[:, 3] - depth <= 0.004 + 1e-7)
    assert not a.diagnostics["used_convex_hull"]
    assert not a.diagnostics["fallback_used"]
    validation = evaluate_mesh_spheres(mesh, a.spheres, request, sample_count=2000)
    assert validation["independent_seed"] != request["seed"]
    assert validation["free_probe_paths"]["opening"]["reference_collision_count"] == 0
    assert validation["free_probe_paths"]["opening"]["sphere_false_positive_count"] == 0
    if validation["collision_miss_count"]:
        assert validation["status"] == "holdout_failed"


@pytest.mark.parametrize("fault", ["budget", "omitted_material", "empty_region", "nonwatertight", "scale", "unknown"])
def test_bad_requests_fail_without_silent_fallback(fault):
    mesh = cup()
    request = config(mesh)
    if fault == "budget":
        request["sphere_budget"] = 8
    elif fault == "omitted_material":
        request["regions"][-1]["bounds_m"][0][0] = 0
    elif fault == "empty_region":
        request["regions"][0]["bounds_m"] = [[1, 1, 1], [2, 2, 2]]
    elif fault == "nonwatertight":
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
    elif fault == "scale":
        mesh.apply_scale(1.2)
    else:
        request["misspelled_radius"] = 1
    with pytest.raises(ValueError):
        generate_mesh_region_spheres(mesh, request)


def test_bad_free_path_is_reported_not_filtered_out():
    mesh = cup()
    request = config(mesh)
    request["free_probe_paths"][0].update(start_m=[0.037, 0, 0.03], end_m=[0.037, 0, 0.07])
    result = generate_mesh_region_spheres(mesh, request)
    report = evaluate_mesh_spheres(mesh, result.spheres, request, sample_count=1000)
    assert report["free_probe_paths"]["opening"]["reference_collision_count"] == 20
    assert report["status"] == "holdout_failed"


def test_giant_cavity_sphere_is_rejected_by_free_space_probes():
    mesh = cup()
    report = evaluate_mesh_spheres(mesh, [[0, 0, 0.05, 0.08]], config(mesh), sample_count=1000)
    assert report["free_probe_paths"]["opening"]["sphere_false_positive_count"] == 20
    assert report["outside_material_center_count"] == 1
    assert report["status"] == "holdout_failed"


def test_input_configuration_is_not_mutated():
    mesh = cup()
    request = config(mesh)
    original = deepcopy(request)
    validate_request(mesh, request)
    generate_mesh_region_spheres(mesh, request)
    assert request == original
