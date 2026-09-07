"""Independent hull-surface, low-dimensional feature and obstacle diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from .hull_structure import area_samples, polyline_samples


def _stats(values):
    from .convex_surface import stats

    return stats(values)


def directions(structure, count=256, hull=None):
    k = np.arange(count)
    z = 1 - 2 * (k + 0.5) / count
    angle = k * np.pi * (3 - np.sqrt(5))
    uniform = np.column_stack(
        (np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z)
    )
    normals = np.array([p["normal"] for p in structure["patches"]])
    axes = np.asarray(structure["box"].get("axes_columns", np.eye(3))).T
    principal = np.linalg.eigh(hull.moment_inertia)[1].T if hull is not None else axes
    return np.vstack((uniform, normals, -normals, axes, -axes, principal, -principal))


def _feature_report(feature, spheres, scale, closed):
    sites, radii = spheres[:, :3], spheres[:, 3]
    points = np.asarray(feature["points_m"])
    sample = polyline_samples(points, 513, closed=closed)
    nearest = cKDTree(sites).query(sample)[0]
    # Assign only sites lying ON this actual polyline, then measure arc gaps.
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    fractions = np.clip(
        np.einsum("nsj,sj->ns", sites[:, None] - points[:-1], delta) / lengths**2, 0, 1
    )
    closest = points[None, :-1] + fractions[:, :, None] * delta
    distances = np.linalg.norm(sites[:, None] - closest, axis=2)
    segment = distances.argmin(axis=1)
    ids = np.flatnonzero(distances[np.arange(len(sites)), segment] <= scale * 1e-5)
    cumulative = np.r_[0, np.cumsum(lengths)]
    arc = np.sort(
        cumulative[segment[ids]] + fractions[ids, segment[ids]] * lengths[segment[ids]]
    )
    if closed:
        gaps = (
            np.diff(np.r_[arc, arc[:1] + cumulative[-1]])
            if len(arc)
            else np.array([cumulative[-1]])
        )
    else:
        gaps = np.diff(np.r_[0, arc, cumulative[-1]])
    nearby = distances[np.arange(len(sites)), segment] <= radii
    # Exact sphere/segment intersections, accumulated in real polyline arc length.
    intervals = []
    for point, radius in zip(sites, radii):
        along = np.einsum("sj,sj->s", point - points[:-1], delta) / lengths**2
        perpendicular2 = (
            np.sum((point - points[:-1]) ** 2, axis=1) - along**2 * lengths**2
        )
        valid = perpendicular2 <= radius**2
        half = np.sqrt(np.maximum(radius**2 - perpendicular2, 0)) / lengths
        left, right = np.maximum(along - half, 0), np.minimum(along + half, 1)
        for j in np.flatnonzero(valid & (left <= right)):
            intervals.append(
                (
                    cumulative[j] + left[j] * lengths[j],
                    cumulative[j] + right[j] * lengths[j],
                )
            )
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + scale * 1e-10:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    uncovered = []
    cursor = 0.0
    for a, b in merged:
        if a > cursor:
            uncovered.append(a - cursor)
        cursor = b
    if cursor < cumulative[-1]:
        uncovered.append(cumulative[-1] - cursor)
    if (
        closed
        and len(uncovered) >= 2
        and merged
        and merged[0][0] > 0
        and merged[-1][1] < cumulative[-1]
    ):
        uncovered = [uncovered[0] + uncovered[-1], *uncovered[1:-1]]
    return {
        "id": feature["id"],
        "length_m": float(cumulative[-1]),
        "site_count_on_feature": len(ids),
        "nearest_site_distance_m": _stats(nearest),
        "arc_gap_m": _stats(gaps),
        "supporting_sphere_count": int(nearby.sum()),
        "exact_sphere_uncovered_arc_gap_m": _stats(uncovered or [0.0]),
        "sphere_covered_arc_fraction": float(
            sum(b - a for a, b in merged) / cumulative[-1]
        ),
        "entire_feature_without_site": not bool(len(ids)),
        "entire_feature_without_sphere_support": not bool(nearby.any()),
        "evaluation_point_count": len(sample),
    }


def evaluate(hull, spheres, structure, *, seed=20260919, sample_count=12000):
    sites, radii = spheres[:, :3], spheres[:, 3]
    query, faces = area_samples(hull, sample_count, seed)
    tree = cKDTree(sites)
    distances, _ = tree.query(query)
    sphere_gap = np.maximum(
        (np.linalg.norm(query[:, None] - sites[None], axis=2) - radii[None]).min(
            axis=1
        ),
        0,
    )
    labels = np.array(structure["face_labels"])[faces]
    patches = []
    for patch in structure["patches"]:
        if patch["area_m2"] < 0.015 * hull.area:
            continue
        selected = distances[labels == patch["id"]]
        patches.append(
            {
                "patch_id": patch["id"],
                "sample_count": len(selected),
                "distance_m": _stats(selected),
            }
        )
    normals = directions(structure, hull=hull)
    h_h = np.max(normals @ hull.vertices.T, axis=1)
    h_s = np.max(normals @ sites.T + radii[None], axis=1)
    error = h_s - h_h
    anchors = np.asarray(structure["corner_anchors_m"]).reshape(-1, 3)
    corners = tree.query(anchors)[0] if len(anchors) else np.empty(0)
    corner_sphere_gap = (
        (np.linalg.norm(anchors[:, None] - sites, axis=2) - radii).min(axis=1)
        if len(anchors)
        else np.empty(0)
    )
    scale = structure["scale_m"]
    return {
        "surface_uniformity": {
            "evaluation_seed": seed,
            "sample_count": sample_count,
            "site_distance_m": _stats(distances),
            "major_patch_distances": patches,
            "sphere_surface_gap_m": _stats(sphere_gap),
            "sphere_surface_gap_mean_m": float(sphere_gap.mean()),
            "sample_fraction_inside_spheres": float(np.mean(sphere_gap == 0)),
        },
        "feature_diagnostics": {
            "corners": {
                "detected": len(anchors),
                "sites_supported": int(np.sum(corners <= scale * 1e-5)),
                "sphere_supported": int(np.sum(corner_sphere_gap <= 0)),
                "nearest_site_distance_m": _stats(corners),
            },
            "edges": [
                _feature_report(e, spheres, scale, e["closed"])
                for e in structure["major_edges"]
            ],
            "rim_loops": [
                _feature_report(e, spheres, scale, True) for e in structure["rim_loops"]
            ],
        },
        "support_error": {
            "direction_count": len(normals),
            "signed_error_m": _stats(error),
            "early_contact_m": _stats(np.maximum(error, 0)),
            "late_contact_m": _stats(np.maximum(-error, 0)),
            "directions": normals.tolist(),
            "error_m": error.tolist(),
            "scope": "exact plane-approach support difference; cannot certify finite/local obstacle detection",
        },
    }


def obstacle_approaches(hull, spheres, structure):
    """Analytic sphere-union vs finite box/cylinder; LP hull contact independently.

    Cylinder hull-reference uses a circumscribed 64-gon, with a reported radial
    error bound. Sphere-cylinder contact is analytic. Translation only; obstacles
    move in -n, their axis is n; no coarse stepping / hidden mesh probe metric.
    """
    scale = structure["scale_m"]
    normals = directions(structure, 12)[:12]
    normals = np.vstack(
        (
            normals,
            np.asarray(structure["box"].get("axes_columns", np.eye(3))).T,
            -np.asarray(structure["box"].get("axes_columns", np.eye(3))).T,
        )
    )
    hn = hull.face_normals
    hb = np.einsum("ij,ij->i", hn, hull.triangles_center)
    results = []
    for direction_id, n in enumerate(normals):
        helper = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        for offset in (0.0, 0.25):
            origin = hull.center_mass + offset * scale * u
            relative = spheres[:, :3] - origin
            a, b, z = relative @ u, relative @ v, relative @ n
            for kind in ("box", "cylinder"):
                width, halfheight = 0.09 * scale, 0.04 * scale
                if kind == "box":
                    on = np.array([u, -u, v, -v, n, -n])
                    bounds = np.array([width] * 4 + [halfheight] * 2)
                    lateral = np.hypot(
                        np.maximum(np.abs(a) - width, 0),
                        np.maximum(np.abs(b) - width, 0),
                    )
                    approximation_error = 0.0
                else:
                    angles = np.arange(64) * 2 * np.pi / 64
                    on = np.vstack(
                        (
                            np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v,
                            n,
                            -n,
                        )
                    )
                    bounds = np.r_[np.full(64, width), halfheight, halfheight]
                    lateral = np.maximum(np.hypot(a, b) - width, 0)
                    approximation_error = float(width * (1 / np.cos(np.pi / 64) - 1))
                # x inside H, x inside obstacle(origin + t*n); maximize t.
                matrix = np.vstack(
                    (
                        np.column_stack((hn, np.zeros(len(hn)))),
                        np.column_stack((on, -on @ n)),
                    )
                )
                rhs = np.r_[hb, bounds + on @ origin]
                solved = linprog(
                    [0, 0, 0, -1],
                    A_ub=matrix,
                    b_ub=rhs,
                    bounds=[(None, None)] * 4,
                    method="highs",
                )
                valid = lateral <= spheres[:, 3]
                t_sphere = (
                    float(
                        np.max(
                            z[valid]
                            + halfheight
                            + np.sqrt(
                                np.maximum(
                                    spheres[valid, 3] ** 2 - lateral[valid] ** 2, 0
                                )
                            )
                        )
                    )
                    if valid.any()
                    else None
                )
                t_hull = float(solved.x[3]) if solved.success else None
                if not solved.success and solved.status != 2:
                    raise RuntimeError(f"Obstacle LP failed: {solved.message}")
                results.append(
                    {
                        "kind": kind,
                        "direction_id": direction_id,
                        "normal": n.tolist(),
                        "lateral_offset_relative": offset,
                        "width_or_radius_m": width,
                        "half_height_m": halfheight,
                        "hull_contact_t_m": t_hull,
                        "sphere_contact_t_m": t_sphere,
                        "signed_contact_error_m": None
                        if t_hull is None or t_sphere is None
                        else t_sphere - t_hull,
                        "hull_hit_but_spheres_miss": t_hull is not None
                        and t_sphere is None,
                        "hull_reference_radial_approximation_bound_m": approximation_error,
                    }
                )
    plane_cases = []
    for n in normals:
        hh = float(np.max(hull.vertices @ n))
        hs = float(np.max(spheres[:, :3] @ n + spheres[:, 3]))
        plane_cases.append(
            {
                "normal": n.tolist(),
                "hull_contact_t_m": hh,
                "sphere_contact_t_m": hs,
                "signed_contact_error_m": hs - hh,
            }
        )
    return {
        "scope": "offline independently solved translation approaches, no Isaac; cylinder hull-reference is bounded 64-gon approximation",
        "plane_cases": plane_cases,
        "cases": results,
        "hull_hits_sphere_misses": sum(r["hull_hit_but_spheres_miss"] for r in results),
        "signed_contact_error_m": _stats(
            [
                r["signed_contact_error_m"]
                for r in results
                if r["signed_contact_error_m"] is not None
            ]
        ),
    }
