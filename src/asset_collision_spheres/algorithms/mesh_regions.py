"""Experimental mesh-region attachment spheres, independent of rendering.

Unlike the analytic handleless-cup specialization, this path consumes the full
watertight collision surface, including handles and protrusions. Explicit local
regions control allocation, not replacement geometry. The legacy manual API is
retained for static diagnostics. The sibling auto_geometry_spheres and
mesh_attachment modules provide the separate automatic and native paths;
precomputed approved manual output can also enter that explicit native adapter.

Centres lie in reference material. A sphere radius is bounded by the centre's
inscribed radius plus an explicit local expansion allowance. Since the inscribed
ball is contained in the material, no sphere point can be farther from material
than that allowance (subject to triangle-query numerical accuracy).
"""

from __future__ import annotations

import hashlib
import json
import time

from ..geometry.types import MeshRegionSphereResult
from ..geometry.material import _closest_point_exact, material_depth, sphere_clearance

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "mesh-region-attachment-spheres-v1-experimental"




def _number(value, name, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} is out of range")
    return float(value)


def _box(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 3) or not np.isfinite(result).all() or np.any(result[1] <= result[0]):
        raise ValueError(f"{name} must contain finite, increasing 3-D bounds")
    return result


def validate_request(mesh, config):
    allowed = {
        "sphere_budget",
        "max_outward_offset_m",
        "expected_bounds_m",
        "bounds_tolerance_m",
        "candidate_sample_count",
        "seed",
        "probe_radius_m",
        "probe_penetration_m",
        "regions",
        "free_probe_paths",
        "description",
        "asset_sha256",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown mesh-region options: {sorted(set(config) - allowed)}")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError(
            "Mesh-region fitting requires a closed, consistently oriented material mesh; no convex-hull fallback"
        )
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces) == 0:
        raise ValueError("Reference mesh is empty or non-finite")
    budget = config["sphere_budget"]
    if type(budget) is not int or not 1 <= budget <= 256:
        raise ValueError("sphere_budget must be an integer in [1, 256]")
    sample_count = config.get("candidate_sample_count", 6000)
    if type(sample_count) is not int or not 500 <= sample_count <= 20000:
        raise ValueError("candidate_sample_count must be an integer in [500, 20000]")
    if type(config.get("seed", 5)) is not int:
        raise ValueError("seed must be an integer")
    tolerance = _number(config.get("bounds_tolerance_m", 2e-6), "bounds_tolerance_m")
    if not np.allclose(mesh.bounds, _box(config["expected_bounds_m"], "expected_bounds_m"), atol=tolerance, rtol=0):
        raise ValueError("Reference bounds changed; update explicit region configuration, do not recenter the mesh")
    _number(config["max_outward_offset_m"], "max_outward_offset_m", positive=False)
    probe = _number(config["probe_radius_m"], "probe_radius_m")
    penetration = _number(config["probe_penetration_m"], "probe_penetration_m")
    if penetration >= probe:
        raise ValueError("probe_penetration_m must be smaller than probe_radius_m")
    regions = config["regions"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("At least one explicit region is required")
    names = set()
    for region in regions:
        if set(region) != {"name", "bounds_m", "sphere_count", "max_radius_m"}:
            raise ValueError("Each region requires name, bounds_m, sphere_count, max_radius_m")
        name = region["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Region names must be unique non-empty strings")
        names.add(name)
        _box(region["bounds_m"], name)
        if type(region["sphere_count"]) is not int or region["sphere_count"] < 1:
            raise ValueError("Each region must receive a positive exact sphere count")
        _number(region["max_radius_m"], "max_radius_m")
    if sum(region["sphere_count"] for region in regions) != budget:
        raise ValueError("Region counts must sum to sphere_budget; no truncation or implicit capacity growth")
    for path in config.get("free_probe_paths", []):
        if set(path) != {"name", "start_m", "end_m", "radius_m", "count"}:
            raise ValueError("Invalid free_probe_paths fields")
        endpoints = np.asarray([path["start_m"], path["end_m"]], dtype=float)
        if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
            raise ValueError("Free-probe endpoints must be finite 3-D points")
        _number(path["radius_m"], "free probe radius")
        if type(path["count"]) is not int or not 2 <= path["count"] <= 1000:
            raise ValueError("Free-probe path count must be in [2, 1000]")
    # Configuration must cover every vertex and triangle centroid, including all
    # extra structures, before a subset is sampled. Overlaps use first-match order.
    classify_regions(np.concatenate([mesh.vertices, mesh.triangles_center]), regions)


def classify_regions(points, regions):
    values = np.asarray(points, dtype=float)
    result = np.full(len(values), -1, dtype=int)
    for index, region in enumerate(regions):
        bounds = _box(region["bounds_m"], region["name"])
        mask = np.all((values >= bounds[0] - 1e-9) & (values <= bounds[1] + 1e-9), axis=1)
        result[(result < 0) & mask] = index
    if np.any(result < 0):
        raise ValueError(f"Region declaration omits {int(np.sum(result < 0))} reference points")
    return result








def _select_region(centres, radii, witnesses, count):
    """Greedy probe coverage within a fixed region quota, then spatial diversity."""
    if len(centres) < count:
        raise ValueError(f"Not enough distinct material candidates for region quota {count}")
    tree = cKDTree(witnesses)
    neighbours = tree.query_ball_point(centres, radii)
    lengths = np.array([len(indices) for indices in neighbours])
    cols = np.concatenate([np.asarray(indices, dtype=int) for indices in neighbours])
    incidence = csr_matrix(
        (np.ones(len(cols)), (np.repeat(np.arange(len(centres)), lengths), cols)), shape=(len(centres), len(witnesses))
    )
    uncovered = np.ones(len(witnesses))
    available = np.ones(len(centres), dtype=bool)
    separation = np.full(len(centres), np.inf)
    selected = []
    for _ in range(count):
        score = np.asarray(incidence @ uncovered).ravel()
        score[~available] = -1
        if score.max() <= 0 and selected:
            score = separation.copy()
            score[~available] = -1
        index = int(np.argmax(score))
        selected.append(index)
        available[index] = False
        uncovered[neighbours[index]] = 0
        separation = np.minimum(separation, np.linalg.norm(centres - centres[index], axis=1))
    return selected


def generate_mesh_region_spheres(mesh, config):
    """Generate finite, explicitly budgeted spheres without replacing the mesh."""
    started = time.perf_counter()
    validate_request(mesh, config)
    surface, faces = trimesh.sample.sample_surface(
        mesh, config.get("candidate_sample_count", 6000), seed=config.get("seed", 5)
    )
    normals = mesh.face_normals[faces]
    # Multiple inward samples find actual material, including both sides of thin
    # walls. No averaging across the empty cup interior and no convex hull.
    offsets = np.array([0.0005, 0.001, 0.002, 0.004, 0.006, 0.008])
    candidates = surface[:, None, :] - normals[:, None, :] * offsets[None, :, None]
    depths = material_depth(mesh, candidates.reshape(-1, 3)).reshape(len(surface), -1)
    best = np.argmax(depths, axis=1)
    centres = candidates[np.arange(len(surface)), best]
    depth = depths[np.arange(len(surface)), best]
    keep = depth > 1e-5
    centres, depth = centres[keep], depth[keep]
    # Quantization is only for duplicate removal; returned coordinates stay exact.
    _, unique = np.unique(np.round(centres / 0.0002).astype(np.int64), axis=0, return_index=True)
    unique.sort()
    centres, depth = centres[unique], depth[unique]
    tags = classify_regions(centres, config["regions"])
    witness_tags = classify_regions(surface, config["regions"])
    probe_radius = config["probe_radius_m"]
    witnesses = surface + normals * (probe_radius - config["probe_penetration_m"])
    selected_spheres, selected_regions, candidates_by_region = [], [], {}
    for index, region in enumerate(config["regions"]):
        mask = tags == index
        local_centres = centres[mask]
        # A small safety subtraction avoids exceeding the explicit expansion
        # allowance because of boundary-query roundoff.
        radii = np.minimum(depth[mask] + config["max_outward_offset_m"] - 1e-6, region["max_radius_m"])
        if np.any(radii <= 0):
            raise ValueError("Expansion allowance leaves non-positive candidate radii")
        local_witnesses = witnesses[witness_tags == index]
        if len(local_witnesses) == 0:
            raise ValueError(f"Region {region['name']} has no surface witnesses")
        picked = _select_region(local_centres, radii + probe_radius, local_witnesses, region["sphere_count"])
        selected_spheres.extend(np.c_[local_centres[picked], radii[picked]].tolist())
        selected_regions.extend([region["name"]] * len(picked))
        candidates_by_region[region["name"]] = int(mask.sum())
    spheres = np.array(selected_spheres)
    result = {
        "algorithm_version": ALGORITHM_VERSION,
        "generation_time_s": time.perf_counter() - started,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "mesh_sha256": hashlib.sha256(
            np.asarray(mesh.vertices, dtype="<f8").tobytes() + np.asarray(mesh.faces, dtype="<i8").tobytes()
        ).hexdigest(),
        "candidate_counts_by_region": candidates_by_region,
        "requested_count": config["sphere_budget"],
        "actual_count": len(spheres),
        "radius_range_m": [float(spheres[:, 3].min()), float(spheres[:, 3].max())],
        "max_outward_offset_m": config["max_outward_offset_m"],
        "sphere_regions": selected_regions,
        "used_convex_hull": False,
        "fallback_used": False,
        "production_adapter_enabled": False,
    }
    return MeshRegionSphereResult(spheres, tuple(selected_regions), result)


def evaluate_mesh_spheres(mesh, spheres, config, *, seed=20260907, sample_count=12000):
    """Hold-out probes on the true mesh; never selects or optimizes spheres."""
    validate_request(mesh, config)
    spheres = np.asarray(spheres, dtype=float)
    if (
        spheres.ndim != 2
        or spheres.shape[1] != 4
        or not len(spheres)
        or not np.isfinite(spheres).all()
        or np.any(spheres[:, 3] <= 0)
    ):
        raise ValueError("Evaluation requires finite positive [x,y,z,r] spheres")
    surface, faces = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    tags = classify_regions(surface, config["regions"])
    probe_radius = config["probe_radius_m"]
    witnesses = surface + mesh.face_normals[faces] * (probe_radius - config["probe_penetration_m"])
    reference_hit = material_depth(mesh, witnesses) >= -probe_radius
    sphere_hit = sphere_clearance(witnesses, spheres, probe_radius) <= 0
    missed = reference_hit & ~sphere_hit
    surface_gap = sphere_clearance(surface, spheres)
    by_region = {}
    for index, region in enumerate(config["regions"]):
        mask = (tags == index) & reference_hit
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Independent hold-out has no probes for {region['name']}")
        by_region[region["name"]] = {
            "reference_collision_count": count,
            "miss_count": int(missed[mask].sum()),
            "detection_fraction": float(sphere_hit[mask].mean()),
        }
    free_paths = {}
    for path in config.get("free_probe_paths", []):
        points = np.linspace(path["start_m"], path["end_m"], path["count"])
        reference_clearance = -material_depth(mesh, points) - path["radius_m"]
        sphere_gap = sphere_clearance(points, spheres, path["radius_m"])
        free_paths[path["name"]] = {
            "probe_radius_m": path["radius_m"],
            "count": path["count"],
            "reference_collision_count": int(np.sum(reference_clearance <= 0)),
            "sphere_false_positive_count": int(np.sum((reference_clearance > 0) & (sphere_gap <= 0))),
            "min_reference_clearance_m": float(reference_clearance.min()),
            "min_sphere_clearance_m": float(sphere_gap.min()),
        }
    centre_depth = material_depth(mesh, spheres[:, :3])
    outside_centres = int(np.sum(centre_depth <= 0))
    expansion_bound = np.maximum(0, spheres[:, 3] - centre_depth)
    invalid_paths = sum(p["reference_collision_count"] for p in free_paths.values())
    false_positives = sum(p["sphere_false_positive_count"] for p in free_paths.values())
    expansion_exceeded = bool(np.any(expansion_bound > config["max_outward_offset_m"] + 2e-6))
    failed = bool(missed.any() or invalid_paths or false_positives or outside_centres or expansion_exceeded)
    return {
        "status": "holdout_failed" if failed else "finite_holdout_passed",
        "scope": "finite mesh probe comparison, not continuous collision proof, PhysX cooking, or planner acceptance",
        "independent_seed": seed,
        "surface_sample_count": sample_count,
        "probe_radius_m": probe_radius,
        "probe_penetration_m": config["probe_penetration_m"],
        "reference_collision_count": int(reference_hit.sum()),
        "collision_miss_count": int(missed.sum()),
        "collision_detection_fraction": float(sphere_hit[reference_hit].mean()),
        "by_region": by_region,
        "free_probe_paths": free_paths,
        "sampled_surface_coverage_fraction": float(np.mean(surface_gap <= 0)),
        "sampled_max_surface_gap_m": float(max(0, surface_gap.max())),
        "outside_material_center_count": outside_centres,
        "max_material_expansion_bound_m": float(expansion_bound.max()),
        "expansion_limit_exceeded": expansion_exceeded,
    }
