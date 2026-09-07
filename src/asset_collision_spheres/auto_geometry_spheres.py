"""Area-sampled full-material fitting with soft, adjacency-geodesic balancing.

No object names, semantic parts, axis assumptions, region AABBs or quotas. The
global ablation shares exactly the same candidate pool and all radius bounds.
Finite diagnostics are evidence, not certification of continuous collision.
"""

from __future__ import annotations

import hashlib
import time
from functools import partial

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .mesh_attachment import (
    characteristic_length,
    check_material_mesh,
    check_spheres,
    mesh_fingerprint,
    sphere_bound_metrics,
)
from .mesh_region_spheres import MeshRegionSphereResult, sphere_clearance
from .mesh_region_spheres import material_depth as _material_depth

material_depth = partial(_material_depth, exact_distance=True)

ALGORITHM_VERSION = "auto-geometry-material-v1-experimental"


def normalize_config(mesh, raw):
    config = dict(raw)
    if "sphere_budget" not in config:
        raise ValueError("An explicit sphere_budget is required (native adapter uses reserved capacity)")
    allowed = {
        "mode",
        "sphere_budget",
        "max_outward_offset_m",
        "seed",
        "candidate_sample_count",
        "patch_count",
        "balance_strength",
        "free_space_weight",
        "redundancy_weight",
        "repair_rounds",
        "probe_radius_m",
        "probe_penetration_m",
    }
    if set(config) - allowed:
        raise ValueError(
            f"Unknown auto geometry options (manual annotations are not inputs): {sorted(set(config) - allowed)}"
        )
    check_material_mesh(mesh)
    config.setdefault("mode", "auto_geometry")
    if config["mode"] not in {"global", "auto_geometry"}:
        raise ValueError("mode must be global or auto_geometry")
    for name, default, low, high in (
        ("sphere_budget", 64, 1, 256),
        ("candidate_sample_count", 6000, 500, 20000),
        ("patch_count", 64, 2, 256),
        ("repair_rounds", 0, 0, 4),
    ):
        config.setdefault(name, default)
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    config.setdefault("seed", 5)
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    scale = characteristic_length(mesh)
    # Global length defaults scale with the mesh, never with a 'cup' category.
    config.setdefault("probe_radius_m", scale * 0.03)
    config.setdefault("probe_penetration_m", config["probe_radius_m"] * 0.25)
    for name, default in (("balance_strength", 1.5), ("free_space_weight", 0.2), ("redundancy_weight", 0.02)):
        config.setdefault(name, default)
    for name in (
        "max_outward_offset_m",
        "probe_radius_m",
        "probe_penetration_m",
        "balance_strength",
        "free_space_weight",
        "redundancy_weight",
    ):
        value = config[name]  # Expansion intentionally has NO universal default.
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < config["probe_penetration_m"] < config["probe_radius_m"]:
        raise ValueError("require 0 < probe_penetration_m < probe_radius_m")
    return config


def local_patches(mesh, sample_faces, count):
    """Voronoi patches on triangle-adjacency graph, not Euclidean through air.

    Dual edge lengths approximate surface distance. Area is measured in m²;
    tessellation influences distance approximation but never supplies a quota.
    """
    adjacent = mesh.face_adjacency
    centroids = mesh.triangles_center
    weights = np.linalg.norm(centroids[adjacent[:, 0]] - centroids[adjacent[:, 1]], axis=1)
    graph = csr_matrix(
        (np.r_[weights, weights], (np.r_[adjacent[:, 0], adjacent[:, 1]], np.r_[adjacent[:, 1], adjacent[:, 0]])),
        shape=(len(mesh.faces), len(mesh.faces)),
    )
    component_count, components = connected_components(graph, directed=False)
    nearest = np.full(len(mesh.faces), np.inf)
    labels = np.full(len(mesh.faces), -1, dtype=int)
    seeds = []
    first = int(sample_faces[0])  # Area-uniform surface sample, not largest face.
    for i in range(min(count, len(mesh.faces))):
        if i == 0:
            seed = first
        elif np.isinf(nearest).any():
            # Prioritize unseen material by component AREA, not triangle count.
            remaining = np.flatnonzero(np.isinf(nearest))
            areas = np.bincount(components[remaining], weights=mesh.area_faces[remaining], minlength=component_count)
            component = int(np.argmax(areas))
            candidates = sample_faces[components[sample_faces] == component]
            seed = int(candidates[0] if len(candidates) else remaining[components[remaining] == component][0])
        else:
            seed = int(np.argmax(nearest))
        distance = dijkstra(graph, directed=False, indices=seed)
        better = distance < nearest
        labels[better] = i
        nearest[better] = distance[better]
        seeds.append(seed)
    # More disconnected pieces than patch budget: keep explicit unassigned
    # component labels rather than merging them through empty space.
    for component in np.unique(components[labels < 0]):
        labels[(components == component) & (labels < 0)] = len(seeds)
        seeds.append(int(np.flatnonzero(components == component)[0]))
    areas = np.bincount(labels, weights=mesh.area_faces)
    return labels, areas, components, seeds


def incidence(centres, radii, points):
    neighbours = cKDTree(points).query_ball_point(centres, radii)
    lengths = np.array([len(row) for row in neighbours])
    columns = np.concatenate([np.asarray(row, dtype=int) for row in neighbours])
    matrix = csr_matrix(
        (np.ones(len(columns)), (np.repeat(np.arange(len(centres)), lengths), columns)),
        shape=(len(centres), len(points)),
    )
    return matrix, neighbours


def candidate_pool(mesh, config):
    scale = characteristic_length(mesh)
    surface, faces = trimesh.sample.sample_surface(mesh, config["candidate_sample_count"], seed=config["seed"])
    normals = mesh.face_normals[faces]
    offsets = scale * 0.04 * np.array([0.00625, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0])
    trials = surface[:, None, :] - normals[:, None, :] * offsets[None, :, None]
    depths = material_depth(mesh, trials.reshape(-1, 3)).reshape(len(surface), -1)
    # Do not keep a later inward sample after observing an air gap on this ray.
    contiguous = np.logical_and.accumulate(depths > scale * 1e-7, axis=1)
    depths = np.where(contiguous, depths, -np.inf)
    # Equal-depth plateaus (e.g. opposing planar faces) must not change their
    # chosen offset due to roundoff after a rigid rotation or OBJ transport.
    ranking_depth = np.round(depths / (scale * 1e-7))
    best = np.argmax(ranking_depth, axis=1)
    depth = depths[np.arange(len(surface)), best]
    keep = np.flatnonzero(depth > scale * 5e-6)
    centres = trials[np.arange(len(surface)), best][keep]
    if len(centres) < config["sphere_budget"]:
        raise ValueError("Too few certifiable material candidates; mesh needs inspection, not hull replacement")
    # Radius-neighbour duplicate removal is rigid-transform invariant; a world
    # axis quantization grid would change candidates when the same mesh rotates.
    tree = cKDTree(centres)
    suppressed = np.zeros(len(centres), dtype=bool)
    unique = []
    for index in range(len(centres)):
        if not suppressed[index]:
            unique.append(index)
            suppressed[tree.query_ball_point(centres[index], scale * 0.001)] = True
    unique = np.asarray(unique)
    origins = keep[unique]
    centres, depth = centres[unique], depth[origins]
    radii = depth + config["max_outward_offset_m"] - scale * 5e-6
    valid = radii > 0
    centres, radii, origins = centres[valid], radii[valid], origins[valid]
    if len(centres) < config["sphere_budget"]:
        raise ValueError("Too few positive-radius distinct candidates for requested budget")
    witnesses = surface + normals * (config["probe_radius_m"] - config["probe_penetration_m"])
    return centres, radii, surface, faces, origins, witnesses


def free_samples(mesh, surface, faces, config, *, seed):
    """Surface-near finite probes, reference-confirmed on the full material mesh.

    Band clearance includes probe radius. The explicitly allowed expansion band
    is reported separately, NOT mislabeled as a hard negative. With exact
    material depth bounds, beyond-band candidate penalties should be zero.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(surface), min(1500, len(surface)), replace=False)
    normal = mesh.face_normals[faces[indices]]
    radius = config["probe_radius_m"]
    expansion = config["max_outward_offset_m"]
    scale = characteristic_length(mesh)
    offsets = radius + np.array([max(expansion * 0.25, scale * 1e-4), max(expansion * 1.25, scale * 0.002)])
    points = (surface[indices, None, :] + normal[:, None, :] * offsets[None, :, None]).reshape(-1, 3)
    clearance = -material_depth(mesh, points) - radius
    confirmed = clearance > scale * 1e-6
    beyond_band = clearance > expansion + scale * 1e-6
    return points, confirmed, beyond_band, radius


def select_greedy(cover, neighbours, witness_tags, patch_areas, free_cost, config, *, initial=(), steps=None):
    selected = list(initial)
    n = cover.shape[1]
    hits = np.zeros(n)
    available = np.ones(cover.shape[0], dtype=bool)
    for index in selected:
        hits[neighbours[index]] += 1
        available[index] = False
    patch_samples = np.bincount(witness_tags, minlength=len(patch_areas))
    # Normalize area influence gently, so a small noisy patch cannot demand a ball.
    area_factor = np.clip(np.sqrt(patch_areas.mean() / np.maximum(patch_areas, 1e-20)), 0.5, 2)
    logs = []
    steps = config["sphere_budget"] - len(selected) if steps is None else steps
    for _ in range(steps):
        uncovered = hits == 0
        covered_by_patch = np.bincount(witness_tags, weights=~uncovered, minlength=len(patch_areas))
        fractions = covered_by_patch / np.maximum(patch_samples, 1)
        boost = config["balance_strength"] * area_factor * (1 - fractions) ** 2
        if config["mode"] == "global":
            boost[:] = 0
        base = np.asarray(cover @ uncovered.astype(float)).ravel() / n
        balance = np.asarray(cover @ (uncovered * boost[witness_tags])).ravel() / n
        redundant = config["redundancy_weight"] * np.asarray(cover @ (~uncovered).astype(float)).ravel() / n
        penalty = config["free_space_weight"] * free_cost
        score = base + balance - redundant - penalty
        score[~available] = -np.inf
        index = int(np.argmax(score))
        selected.append(index)
        available[index] = False
        hits[neighbours[index]] += 1
        logs.append(
            {
                "candidate": index,
                "contact_gain": float(base[index]),
                "local_balance_gain": float(balance[index]),
                "redundancy_penalty": float(redundant[index]),
                "free_space_penalty": float(penalty[index]),
                "total": float(score[index]),
            }
        )
    return selected, logs


def _objective(selected, neighbours, tags, area, free_cost, config):
    hit = np.zeros(len(tags), dtype=bool)
    for index in selected:
        hit[neighbours[index]] = True
    counts = np.bincount(tags, minlength=len(area))
    fraction = np.bincount(tags, weights=hit, minlength=len(area)) / np.maximum(counts, 1)
    balance = config["balance_strength"] if config["mode"] == "auto_geometry" else 0
    return float(
        hit.mean()
        + balance * np.sum(area * np.sqrt(fraction)) / area.sum()
        - config["free_space_weight"] * free_cost[selected].sum()
    )


def evaluate_auto_geometry_spheres(mesh, spheres, config, *, seed=20260908, sample_count=12000):
    """Independent seed; ordinary misses never become automatic rejection."""
    config = normalize_config(mesh, config)
    depth = check_spheres(mesh, spheres, config["sphere_budget"], config["max_outward_offset_m"])
    spheres = np.asarray(spheres)
    surface, faces = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    radius = config["probe_radius_m"]
    witnesses = surface + mesh.face_normals[faces] * (radius - config["probe_penetration_m"])
    reference_hit = material_depth(mesh, witnesses) >= -radius
    detected = sphere_clearance(witnesses, spheres, radius) <= 0
    points, confirmed, beyond_band, free_radius = free_samples(mesh, surface, faces, config, seed=seed + 1)
    false = sphere_clearance(points, spheres, free_radius) < 0
    gap = np.maximum(sphere_clearance(surface, spheres), 0)
    return {
        "independent_seed": seed,
        "ordinary_sample_count": sample_count,
        "contact_probe_count": int(reference_hit.sum()),
        "ordinary_contact_misses": int((reference_hit & ~detected).sum()),
        "contact_detection_fraction": float(detected[reference_hit].mean()),
        "surface_coverage": float(np.mean(gap <= 1e-8)),
        "surface_gap_mean_m": float(gap.mean()),
        "surface_gap_p95_m": float(np.quantile(gap, 0.95)),
        "max_uncovered_gap_m": float(gap.max()),
        "free_probe_count_beyond_allowance": int(beyond_band.sum()),
        "free_probe_intrusions_beyond_allowance": int((false & beyond_band).sum()),
        "free_probe_count_in_allowed_band": int((confirmed & ~beyond_band).sum()),
        "allowed_band_occupancy_count_not_failure": int((false & confirmed & ~beyond_band).sum()),
        "max_radius_minus_material_depth_m": float(np.max(spheres[:, 3] - depth)),
        "acceptance": "finite_checks_passed; ordinary gaps diagnostic; inspect major missing structures and cavities",
    }


def generate_auto_geometry_spheres(mesh, raw):
    started = time.perf_counter()
    config = normalize_config(mesh, raw)
    candidate_start = time.perf_counter()
    centres, radii, surface, faces, origins, witnesses = candidate_pool(mesh, config)
    candidate_time = time.perf_counter() - candidate_start
    patch_start = time.perf_counter()
    labels, area, components, _seeds = local_patches(mesh, faces, config["patch_count"])
    tags = labels[faces]
    cover, neighbours = incidence(centres, radii + config["probe_radius_m"], witnesses)
    free_points, _, free_mask, probe_radius = free_samples(mesh, surface, faces, config, seed=config["seed"] + 31)
    rng = np.random.default_rng(config["seed"] + 37)
    invasion_ids = rng.choice(len(centres), min(512, len(centres)), replace=False)
    directions = mesh.face_normals[faces[rng.integers(0, len(faces), len(invasion_ids))]]
    invasion_points = centres[invasion_ids] + directions * (radii[invasion_ids, None] + probe_radius) * 0.95
    invasion_clearance = -material_depth(mesh, invasion_points) - probe_radius
    invasion_mask = invasion_clearance > config["max_outward_offset_m"] + characteristic_length(mesh) * 1e-6
    free_points = np.r_[free_points, invasion_points]
    free_mask = np.r_[free_mask, invasion_mask]
    if free_mask.any():
        free_cover, _ = incidence(centres, radii + probe_radius, free_points[free_mask])
        free_cost = np.asarray(free_cover.sum(axis=1)).ravel() / free_mask.sum()
    else:
        free_cost = np.zeros(len(centres))
    graph_and_score_time = time.perf_counter() - patch_start
    select_start = time.perf_counter()
    selected, trace = select_greedy(cover, neighbours, tags, area, free_cost, config)
    select_time = time.perf_counter() - select_start
    repair_start = time.perf_counter()
    repair_log = []
    # Strictly bounded same-budget swap, at most one accepted swap per round.
    for iteration in range(config["repair_rounds"]):
        old_score = _objective(selected, neighbours, tags, area, free_cost, config)
        removals = [
            _objective(selected[:i] + selected[i + 1 :], neighbours, tags, area, free_cost, config)
            for i in range(len(selected))
        ]
        removed = int(np.argmax(removals))
        proposal, _ = select_greedy(
            cover,
            neighbours,
            tags,
            area,
            free_cost,
            config,
            initial=selected[:removed] + selected[removed + 1 :],
            steps=1,
        )
        new_score = _objective(proposal, neighbours, tags, area, free_cost, config)
        improved = new_score > old_score + 1e-8
        repair_log.append({"round": iteration, "old": old_score, "new": new_score, "accepted": improved})
        if not improved:
            break
        selected = proposal
    repair_time = time.perf_counter() - repair_start
    spheres = np.c_[centres[selected], radii[selected]]
    selected_depths = check_spheres(mesh, spheres, config["sphere_budget"], config["max_outward_offset_m"])
    candidate_digest = hashlib.sha256(np.c_[centres, radii].astype("<f8").tobytes()).hexdigest()
    hit = np.zeros(len(tags), dtype=bool)
    for index in selected:
        hit[neighbours[index]] = True
    counts = np.bincount(tags, minlength=len(area))
    patch_coverage = np.bincount(tags, weights=hit, minlength=len(area)) / np.maximum(counts, 1)
    missing_components = sorted(set(components.tolist()) - set(components[faces[origins[selected]]].tolist()))
    patches = [
        {
            "id": i,
            "area_m2": float(a),
            "sample_count": int(counts[i]),
            "contact_coverage": float(patch_coverage[i]),
            "selected_origin_count": int(np.sum(tags[origins[selected]] == i)),
        }
        for i, a in enumerate(area)
    ]
    generation_time = time.perf_counter() - started
    diagnostic_start = time.perf_counter()
    evaluation = evaluate_auto_geometry_spheres(mesh, spheres, config)
    diagnostic_time = time.perf_counter() - diagnostic_start
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "mode": config["mode"],
        "config": config,
        "mesh_sha256": mesh_fingerprint(mesh),
        "candidate_pool_sha256": candidate_digest,
        "candidate_count": len(centres),
        "actual_count": len(spheres),
        "max_outward_offset_m": config["max_outward_offset_m"],
        "radius_range_m": [float(radii[selected].min()), float(radii[selected].max())],
        "generation_time_s": generation_time,
        "candidate_time_s": candidate_time,
        "patch_and_incidence_time_s": graph_and_score_time,
        "selection_time_s": select_time,
        "repair_time_s": repair_time,
        "diagnostic_time_s": diagnostic_time,
        "repair_log": repair_log,
        "selection_trace": trace,
        "patches": patches,
        "missing_material_components": missing_components,
        "material_assumptions": (
            "closed consistently oriented non-self-intersecting material boundary; "
            "watertightness/winding/degeneracy checked, exhaustive self-intersection detection not implemented"
        ),
        "review_required": bool(missing_components),
        "zero_contact_patch_ids": [i for i in range(len(area)) if counts[i] and patch_coverage[i] == 0],
        "patch_definition": (
            "area-weighted diagnostics; face-adjacency dual-graph approximate geodesic, no semantic labels"
        ),
        "free_penalty_candidate_nonzero_count": int(np.count_nonzero(free_cost)),
        "candidate_expansion_free_samples": len(invasion_points),
        "free_penalty_note": (
            "Exact depth+allowance bound implies zero beyond-band intrusion; allowed band is not penalized"
        ),
        "used_convex_hull": False,
        "fallback_used": False,
        "validation_status": "finite_checks_passed",
        "diagnostics": evaluation,
        **{
            k: evaluation[k]
            for k in (
                "surface_coverage",
                "surface_gap_mean_m",
                "surface_gap_p95_m",
                "max_uncovered_gap_m",
                "ordinary_sample_count",
            )
        },
        **sphere_bound_metrics(mesh, spheres, selected_depths),
    }
    return MeshRegionSphereResult(spheres, tuple(f"patch_{tags[origins[index]]:03d}" for index in selected), report)
