"""Independent convex exterior surface sites; bounded shared radii, <=64 spheres.

The hull, not original material, is the optimization and diagnostic reference.
Internal sampling variants exist only as Python experiment arguments.
"""

from __future__ import annotations

from itertools import product
import time

import numpy as np
from scipy.spatial import cKDTree

from .hull_structure import (
    analyze,
    area_samples,
    convex_hull,
    polyline_samples,
    project,
)
from ..geometry.types import MeshRegionSphereResult
from ..geometry.validation import mesh_fingerprint

VERSION = "convex-surface-v3-experimental"
SYSTEM_MAX_SPHERES = 64
REFERENCE_COUNT = 8192
EVALUATION_COUNT = 12000
EVALUATION_SEED = 20260919
POLICY = {
    "radius_small_scale": 0.10,
    "radius_base_m": 0.006,
    "radius_scale": 0.02,
    "radius_absolute_max_m": 0.012,
    "feature_fraction": 0.45,
    "feature_spacing_ratio": 0.8,
    "proxy_error_relative": 0.08,
    "reference_count": REFERENCE_COUNT,
    "evaluation_count": EVALUATION_COUNT,
    "generic_stop": "capacity (no calibrated early-stop threshold yet)",
}


def normalize(raw=None):
    raw = dict(raw or {})
    allowed = {"mode", "shape_hint", "max_spheres", "seed", "backend_sphere_capacity"}
    if set(raw) - allowed:
        raise ValueError(
            f"Unknown convex_surface settings: {sorted(set(raw) - allowed)}"
        )
    if raw.get("mode", "convex_surface") != "convex_surface":
        raise ValueError("mode must be convex_surface")
    hint = raw.get("shape_hint", "auto")
    if hint not in {"auto", "open_box", "generic"}:
        raise ValueError("shape_hint must be auto, open_box or generic")
    cap = raw.get("max_spheres", 64)
    if type(cap) is not int or not 1 <= cap <= 64:
        raise ValueError("convex_surface max_spheres must be an integer in [1,64]")
    backend = raw.get("backend_sphere_capacity", 64)
    if type(backend) is not int or backend < 1:
        raise ValueError("backend_sphere_capacity must be a positive integer")
    seed = raw.get("seed", 5)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {
        "mode": "convex_surface",
        "shape_hint": hint,
        "max_spheres": cap,
        "backend_sphere_capacity": backend,
        "seed": seed,
        "effective_max_spheres": min(cap, backend, 64),
    }


def stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return dict(
        zip(
            ("min", "p25", "median", "p75", "p95", "max"),
            map(float, np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1])),
        )
    )


def grid_count(nodes):
    return int(np.prod(nodes) - np.prod(np.maximum(np.asarray(nodes) - 2, 0)))


def choose_grid(lengths, cap):
    if cap < 8:
        return None
    lengths = np.asarray(lengths)
    best = None
    # A 2 x 2 x n boundary consumes 4n sites, so this is exhaustive <=64.
    for nodes in product(range(2, cap // 4 + 1), repeat=3):
        count = grid_count(nodes)
        if count > cap:
            continue
        spacing = lengths / (np.asarray(nodes) - 1)
        isotropy = float(np.std(np.log(spacing)))
        unused = (cap - count) / cap
        # Relative lengths give longer axes extra nodes naturally. A small
        # utilization term favors refinement without sacrificing regularity.
        score = isotropy + 0.28 * unused
        if best is None or score < best[0] - 1e-12:
            best = (score, nodes)
    return best[1]


def _unique_add(sites, kinds, point, kind, tolerance):
    if not sites or np.linalg.norm(np.asarray(sites) - point, axis=1).min() > tolerance:
        sites.append(np.asarray(point))
        kinds.append(kind)
        return True
    return False


def box_grid(hull, structure, cap):
    box = structure["box"]
    nodes = choose_grid(box["lengths_m"], cap)
    if nodes is None:
        return (
            [],
            [],
            {"reason": "capacity below eight box corners", "review_required": True},
        )
    axes, origin = np.asarray(box["axes_columns"]), np.asarray(box["origin_m"])
    coordinates = [
        np.linspace(a, b, n) for a, b, n in zip(box["low_m"], box["high_m"], nodes)
    ]
    targets, sources = [], []
    for index in product(*(range(n) for n in nodes)):
        on_boundary = sum(i in (0, n - 1) for i, n in zip(index, nodes))
        if on_boundary:
            point = np.array([coordinates[d][i] for d, i in enumerate(index)])
            targets.append(origin + axes @ point)
            sources.append({1: "face", 2: "edge", 3: "corner"}[on_boundary])
    mapped, errors, _ = project(hull, targets)
    sites, kinds = [], []
    replaced = []
    for i, (point, kind, error) in enumerate(zip(mapped, sources, errors)):
        if error > POLICY["proxy_error_relative"] * structure["scale_m"]:
            replaced.append(i)
            continue
        _unique_add(sites, kinds, point, kind, structure["scale_m"] * 1e-6)
    return (
        sites,
        kinds,
        {
            "nodes_xyz": list(nodes),
            "unique_boundary_target_count": len(targets),
            "nominal_spacing_m": (
                np.asarray(box["lengths_m"]) / (np.asarray(nodes) - 1)
            ).tolist(),
            "proxy_mapping_error_m": stats(errors),
            "replaced_by_generic_target_ids": replaced,
            "proxy_targets_m": np.asarray(targets).tolist(),
            "deduplicated_count": len(targets) - len(replaced) - len(sites),
            "review_required": bool(replaced),
        },
    )


def feature_seeds(hull, structure, cap):
    sites, kinds = [], []
    limit = int(np.floor(cap * POLICY["feature_fraction"]))
    scale = structure["scale_m"]
    spacing = np.sqrt(hull.area / cap) * POLICY["feature_spacing_ratio"]
    loops = sorted(structure["rim_loops"], key=lambda p: -p["patch_area_m2"])
    # Allocate by length/spacing; scale all requests together if features exceed
    # their shared fraction. At least 55% remains for ordinary surface sites.
    desired = np.array(
        [max(3, int(np.ceil(p["length_m"] / spacing))) for p in loops], int
    )
    allocations = desired.copy()
    if desired.sum() > limit:
        exact = desired * limit / desired.sum()
        allocations = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - allocations), kind="stable")[
            : limit - allocations.sum()
        ]:
            allocations[i] += 1
    allocation_report = []
    for loop, count in zip(loops, allocations):
        # Two points don't meaningfully represent a loop: report it as omitted.
        count = int(count) if count >= 3 else 0
        before = len(sites)
        for point in polyline_samples(loop["points_m"], count, closed=True):
            _unique_add(sites, kinds, point, "rim", scale * 1e-5)
        allocation_report.append(
            {
                "loop_id": loop["id"],
                "requested_from_spacing": int(desired[len(allocation_report)]),
                "assigned_unique_sites": len(sites) - before,
            }
        )
    # Remaining feature capacity can represent stable true corners / long edges.
    for point in structure["corner_anchors_m"]:
        if len(sites) >= limit:
            break
        _unique_add(sites, kinds, point, "corner", scale * 1e-5)
    for edge in sorted(structure["major_edges"], key=lambda p: -p["length_m"]):
        count = max(2, int(np.ceil(edge["length_m"] / spacing)) + 1)
        for point in polyline_samples(edge["points_m"], count):
            if len(sites) >= limit:
                break
            if (
                sites
                and np.linalg.norm(np.asarray(sites) - point, axis=1).min()
                < spacing * 0.45
            ):
                continue
            _unique_add(sites, kinds, point, "edge", scale * 1e-5)
    return (
        sites,
        kinds,
        {
            "feature_limit": limit,
            "target_feature_spacing_m": float(spacing),
            "loop_allocations": allocation_report,
            "actual_feature_seeds": len(sites),
        },
    )


def fps(reference, sites, kinds, target, *, kind="generic"):
    curve = []
    if not sites:
        # Stable extremum relative to the geometry's own centroid, not world XYZ.
        center = reference.mean(axis=0)
        index = int(np.argmax(np.linalg.norm(reference - center, axis=1)))
        sites.append(reference[index].copy())
        kinds.append(kind)
    nearest = cKDTree(sites).query(reference)[0]
    while True:
        if len(sites) % 4 == 0 or len(sites) == target:
            curve.append({"count": len(sites), **stats(nearest)})
        if len(sites) >= target:
            break
        index = int(np.argmax(nearest))
        if nearest[index] <= 1e-12:
            break
        point = reference[index]
        sites.append(point.copy())
        kinds.append(kind)
        nearest = np.minimum(nearest, np.linalg.norm(reference - point, axis=1))
    return curve


def shared_radius(hull):
    # Hull-area equivalent length, independent of sphere count and semantic class.
    # 12 mm absolute cap prevents large assets producing giant collision balls.
    length = float(np.sqrt(hull.area / (2 * np.pi)))
    return min(
        POLICY["radius_small_scale"] * length,
        POLICY["radius_base_m"] + POLICY["radius_scale"] * length,
        POLICY["radius_absolute_max_m"],
    )


def check_convex_spheres(hull, spheres, cap):
    spheres = np.asarray(spheres, dtype=float)
    if (
        spheres.ndim != 2
        or spheres.shape[1] != 4
        or not 1 <= len(spheres) <= min(cap, 64)
    ):
        raise ValueError(
            "Convex spheres must have shape [N,4], 1 <= N <= capacity <=64"
        )
    if (
        not np.isfinite(spheres).all()
        or np.any(spheres[:, 3] <= 0)
        or np.any(spheres[:, 3] > 0.012 + 1e-12)
    ):
        raise ValueError("Invalid convex sphere radius/coordinates")
    # A convex polytope's boundary is inside all halfspaces and on at least one
    # support plane. This also avoids closest-triangle roundoff on scan slivers.
    offsets = np.einsum("ij,ij->i", hull.face_normals, hull.triangles_center)
    signed = spheres[:, :3] @ hull.face_normals.T - offsets
    distance = np.abs(signed.max(axis=1))
    tolerance = max(1e-9, np.sqrt(hull.area) * 1e-7)
    if distance.max() > tolerance:
        raise ValueError("Convex sampling sites must lie on the actual hull surface")
    if (
        len(spheres) > 1
        and cKDTree(spheres[:, :3]).query(spheres[:, :3], k=2)[0][:, 1].min()
        <= tolerance
    ):
        raise ValueError("Duplicate convex sampling sites")
    return float(distance.max())


def generate_convex_surface_spheres(mesh, raw=None, *, _variant="auto", _count=None):
    """Public mesh + tiny config API. Underscored arguments are ablation-only."""
    started = time.perf_counter()
    config = normalize(raw)
    hull = convex_hull(mesh)
    structure = analyze(hull, config["shape_hint"])
    cap = config["effective_max_spheres"]
    if _count is not None:
        if type(_count) is not int or not 1 <= _count <= cap:
            raise ValueError("Internal ablation count must fit the normal hard cap")
        target = _count
    else:
        target = cap
    if _variant not in {"auto", "random", "fps", "features", "grid"}:
        raise ValueError("Unknown internal sampling ablation")
    variant = (
        ("grid" if structure["box"]["enabled"] and cap >= 8 else "features")
        if _variant == "auto"
        else _variant
    )
    reference, _ = area_samples(hull, REFERENCE_COUNT, config["seed"])
    # Actual vertices supplement the area reference so extrema aren't lost.
    reference = np.vstack((reference, hull.vertices))
    selection = {}
    if variant == "grid":
        if not structure["box"]["enabled"]:
            raise ValueError("Grid ablation requires supported box-like hull")
        sites, kinds, selection = box_grid(hull, structure, target)
        if not sites:
            variant = "features"
        else:
            target = selection["unique_boundary_target_count"]
    if variant == "features":
        sites, kinds, selection = feature_seeds(hull, structure, target)
    elif variant == "fps":
        sites, kinds = [], []
    elif variant == "random":
        import trimesh

        points, _ = trimesh.sample.sample_surface(hull, target, seed=config["seed"])
        sites, kinds = list(points), ["random"] * target
    if variant in {"fps", "features"}:
        # Six geometry-frame support extrema prevent purely area-based samples
        # from neglecting the outer silhouette. These use ordinary capacity.
        _, axes = np.linalg.eigh(hull.moment_inertia)
        extrema = []
        for axis in axes.T:
            for sign in (-1, 1):
                extrema.append(hull.vertices[np.argmax(hull.vertices @ (sign * axis))])
        for point in extrema:
            if len(sites) >= target:
                break
            _unique_add(sites, kinds, point, "generic", structure["scale_m"] * 1e-5)
    curve = fps(reference, sites, kinds, target)
    sites = np.asarray(sites)
    radius = shared_radius(hull)
    spheres = np.column_stack((sites, np.full(len(sites), radius)))
    surface_error = check_convex_spheres(hull, spheres, cap)
    from .hull_diagnostics import evaluate

    diagnostics = evaluate(hull, spheres, structure)
    diagnostics.update(
        {
            "algorithm_version": VERSION,
            "mode": "convex_surface",
            "target_geometry": "convex_hull",
            "config": config,
            "system_policy": POLICY,
            "actual_count": len(spheres),
            "effective_max_spheres": cap,
            "sampling_strategy": variant,
            "count_stop_reason": "regular_boundary_grid"
            if variant == "grid"
            else "effective_capacity",
            "radius_m": stats(spheres[:, 3]),
            "radius_policy": "L=sqrt(hull.area/(2*pi)); r=min(0.10*L, 0.006 m+0.02*L, 0.012 m)",
            "surface_site_max_error_m": surface_error,
            "selection": selection,
            "surface_site_validation": "all hull halfspaces + at least one support plane; exact for a convex polytope within numerical tolerance",
            "incremental_reference_distance_m": curve,
            "hull_structure": structure,
            "hull_vertices_m": hull.vertices.tolist(),
            "hull_faces": hull.faces.tolist(),
            "original_mesh_sha256": mesh_fingerprint(mesh),
            "hull_sha256": mesh_fingerprint(hull),
            "hull_area_m2": float(hull.area),
            "hull_volume_m3": float(hull.volume),
            "sphere_sources": kinds,
            "validation_status": "finite_hull_geometry_checks_passed",
            "review_required": bool(
                selection.get("review_required")
                or (structure["box"]["enabled"] and cap < 8)
            ),
            "acceptance": "sparse hull-exterior approximation; not local collision completeness or a trajectory safety certificate",
            "generation_time_s": time.perf_counter() - started,
        }
    )
    return MeshRegionSphereResult(spheres, tuple(kinds), diagnostics)
