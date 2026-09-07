"""Convex-hull geometry only: planes, real boundaries and supported box frames.

No material membership, cavity probes, semantic labels or world-axis box tests.
All geometric thresholds below are shared angles or relative hull distances.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import trimesh


def convex_hull(mesh):
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("Convex surface needs at least four 3D vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("Convex surface vertices must be finite")
    vertices = np.unique(vertices, axis=0)
    origin = vertices.mean(axis=0)
    scale = float(np.max(np.linalg.norm(vertices - origin, axis=1)))
    if not np.isfinite(scale) or scale < 1e-9:
        raise ValueError("Degenerate convex-hull scale")
    normalized = (vertices - origin) / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    if len(singular) < 3 or singular[-1] < singular[0] * 1e-6:
        raise ValueError("Coplanar or near-degenerate convex-hull point set")
    try:
        qh = ConvexHull(normalized)  # Deliberately no QJ jitter / fabricated thickness.
    except QhullError as error:
        raise ValueError(
            "Convex hull construction failed; no geometry fabricated"
        ) from error
    faces = qh.simplices.copy()
    triangles = normalized[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    flip = np.einsum("ij,ij->i", cross, qh.equations[:, :3]) < 0
    faces[flip] = faces[flip, ::-1]
    hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hull.remove_unreferenced_vertices()
    if not hull.is_volume or hull.volume < scale**3 * 1e-9:
        raise ValueError("Hull is not a stable positive-volume solid")
    return hull


def area_samples(hull, count, seed):
    """Stratified area allocation + deterministic barycentric samples, not greedy candidates."""
    rng = np.random.default_rng(seed)
    allocation = hull.area_faces / hull.area * count
    counts = np.floor(allocation).astype(int)
    ids = np.argsort(-(allocation - counts), kind="stable")[: count - counts.sum()]
    counts[ids] += 1
    faces = np.repeat(np.arange(len(counts)), counts)
    # Equal-area transformation of uniform pairs within each triangle.
    uv = rng.random((count, 2))
    a = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - a, a * (1 - uv[:, 1]), a * uv[:, 1]))
    points = np.einsum("ni,nij->nj", bary, hull.triangles[faces])
    return points, faces


def project(hull, points):
    """Nearest points on actual hull triangles (never on a proxy box)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return points.copy(), np.empty(0), np.empty(0, int)
    return trimesh.proximity.closest_point(hull, points)


def polyline_samples(points, count, closed=False):
    points = np.asarray(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    positions = np.linspace(0, cumulative[-1], count, endpoint=not closed)
    indices = np.minimum(
        np.searchsorted(cumulative, positions, side="right") - 1, len(lengths) - 1
    )
    fractions = (positions - cumulative[indices]) / np.maximum(lengths[indices], 1e-30)
    return points[indices] + fractions[:, None] * (
        points[indices + 1] - points[indices]
    )


def _chains(edges):
    """Decompose undirected boundary graph into maximal degree-two chains / loops."""
    remaining = {tuple(sorted(map(int, edge))) for edge in edges}
    graph = {}
    for a, b in remaining:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    chains = []
    while remaining:
        endpoints = sorted(v for e in remaining for v in e if len(graph[v]) != 2)
        start = endpoints[0] if endpoints else min(min(e) for e in remaining)
        chain = [start]
        while True:
            options = sorted(
                v
                for v in graph[chain[-1]]
                if tuple(sorted((v, chain[-1]))) in remaining
            )
            if not options:
                break
            other = options[0]
            remaining.remove(tuple(sorted((other, chain[-1]))))
            chain.append(other)
            if other == start or len(graph[other]) != 2:
                break
        if len(chain) > 1:
            chains.append(chain)
    return chains


def analyze(hull, hint="auto"):
    scale = float(np.sqrt(hull.area / (2 * np.pi)))
    normals, areas, centers = hull.face_normals, hull.area_faces, hull.triangles_center
    adjacency, adjacent_edges = hull.face_adjacency, hull.face_adjacency_edges
    graph = [[] for _ in areas]
    for a, b in adjacency:
        graph[a].append(int(b))
        graph[b].append(int(a))
    labels = np.full(len(areas), -1, int)
    patches = []
    # Seed-referenced (not just transitive pairwise) growth prevents drifting
    # around a smooth bowl and calling the entire curved surface one plane.
    for seed in np.argsort(-areas, kind="stable"):
        if labels[seed] >= 0:
            continue
        queue, seen, accepted = [int(seed)], {int(seed)}, []
        while queue:
            f = queue.pop()
            if labels[f] >= 0 or normals[f] @ normals[seed] < np.cos(np.deg2rad(10)):
                continue
            if (
                np.max(np.abs((hull.triangles[f] - centers[seed]) @ normals[seed]))
                > 0.004 * scale
            ):
                continue
            labels[f] = len(patches)
            accepted.append(f)
            for other in graph[f]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        ids = np.asarray(accepted)
        normal = np.average(normals[ids], axis=0, weights=areas[ids])
        normal /= np.linalg.norm(normal)
        center = np.average(centers[ids], axis=0, weights=areas[ids])
        patches.append(
            {
                "id": len(patches),
                "normal": normal.tolist(),
                "center_m": center.tolist(),
                "face_ids": ids.tolist(),
                "area_m2": float(areas[ids].sum()),
                "residual_max_m": float(
                    np.abs((hull.triangles[ids] - center) @ normal).max()
                ),
                "adjacent_patch_ids": [],
            }
        )
    # Merge adjacent near-coplanar seed patches with a whole-group residual and
    # angular guard. This removes scan-induced splits across a broad hull cap.
    groups = [{i} for i in range(len(patches))]
    owners = np.arange(len(patches))
    for a, b in adjacency:
        p, q = int(owners[labels[a]]), int(owners[labels[b]])
        if p == q:
            continue
        members = groups[p] | groups[q]
        ids = np.array([f for pid in sorted(members) for f in patches[pid]["face_ids"]])
        normal = np.average(normals[ids], axis=0, weights=areas[ids])
        normal /= np.linalg.norm(normal)
        center = np.average(centers[ids], axis=0, weights=areas[ids])
        if np.min(normals[ids] @ normal) < np.cos(np.deg2rad(12)):
            continue
        if np.abs((hull.triangles[ids] - center) @ normal).max() > 0.012 * scale:
            continue
        groups[p] = members
        groups[q] = set()
        for pid in members:
            owners[pid] = p
    merged = []
    for members in groups:
        if not members:
            continue
        ids = np.array(sorted(f for pid in members for f in patches[pid]["face_ids"]))
        normal = np.average(normals[ids], axis=0, weights=areas[ids])
        normal /= np.linalg.norm(normal)
        center = np.average(centers[ids], axis=0, weights=areas[ids])
        labels[ids] = len(merged)
        merged.append(
            {
                "id": len(merged),
                "normal": normal.tolist(),
                "center_m": center.tolist(),
                "face_ids": ids.tolist(),
                "area_m2": float(areas[ids].sum()),
                "residual_max_m": float(
                    np.abs((hull.triangles[ids] - center) @ normal).max()
                ),
                "adjacent_patch_ids": [],
            }
        )
    patches = merged
    pairs = {}
    boundaries = [[] for _ in patches]
    sharp = []
    for (a, b), edge in zip(adjacency, adjacent_edges):
        p, q = int(labels[a]), int(labels[b])
        if p == q:
            continue  # Internal triangulation diagonal is NEVER a feature.
        angle = float(np.arccos(np.clip(normals[a] @ normals[b], -1, 1)))
        pairs.setdefault(tuple(sorted((p, q))), []).append((edge, angle))
        boundaries[p].append((edge, angle))
        boundaries[q].append((edge, angle))
        if angle >= np.deg2rad(28):
            sharp.append(edge)
    for p, q in pairs:
        patches[p]["adjacent_patch_ids"].append(q)
        patches[q]["adjacent_patch_ids"].append(p)
    edges = []
    # Group boundaries between the same patches before chaining; this merges
    # collinear fragments while keeping intersections as shared endpoints.
    for (p, q), values in pairs.items():
        qualifying = [edge for edge, angle in values if angle >= np.deg2rad(28)]
        for chain in _chains(qualifying):
            points = hull.vertices[chain]
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length >= 0.065 * scale:
                edges.append(
                    {
                        "id": len(edges),
                        "patch_ids": [p, q],
                        "points_m": points.tolist(),
                        "length_m": length,
                        "closed": chain[0] == chain[-1],
                    }
                )
    loops = []
    for patch, values in zip(patches, boundaries):
        if patch["area_m2"] < 0.035 * hull.area or not values:
            continue
        # Require actual dihedral evidence along a substantial boundary length.
        weights = np.array(
            [
                np.linalg.norm(hull.vertices[e[0]] - hull.vertices[e[1]])
                for e, _ in values
            ]
        )
        patch_normal = np.asarray(patch["normal"])
        evidence = []
        for edge, angle in values:
            # Scan bevels distribute turning across several small facets. Look
            # through a bounded local surface band, not only one triangle pair.
            midpoint = hull.vertices[edge].mean(axis=0)
            nearby = np.linalg.norm(centers - midpoint, axis=1) <= 0.065 * scale
            turning = np.arccos(np.clip(normals[nearby] @ patch_normal, -1, 1))
            evidence.append(
                angle >= np.deg2rad(15)
                or (len(turning) and np.max(turning) >= np.deg2rad(25))
            )
        sharp_fraction = float(np.sum(weights * np.array(evidence)) / weights.sum())
        if sharp_fraction < 0.55:
            continue
        for chain in _chains([e for e, _ in values]):
            if chain[0] != chain[-1] or len(chain) < 4:
                continue
            points = hull.vertices[chain]
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length < 0.4 * scale:
                continue
            # Deduplicate coincident boundaries (e.g. two sides of a very thin hull).
            signature = set(chain)
            if any(signature == set(loop["vertex_ids"]) for loop in loops):
                continue
            loops.append(
                {
                    "id": len(loops),
                    "patch_id": patch["id"],
                    "points_m": points.tolist(),
                    "vertex_ids": chain,
                    "length_m": length,
                    "closed": True,
                    "sharp_boundary_fraction": sharp_fraction,
                    "patch_area_m2": patch["area_m2"],
                }
            )
    # Three distinct sharp directions incident at a real hull vertex => corner.
    incident = {}
    for a, b in sharp:
        incident.setdefault(int(a), []).append(int(b))
        incident.setdefault(int(b), []).append(int(a))
    corners = []
    for v, neighbors in incident.items():
        directions = hull.vertices[neighbors] - hull.vertices[v]
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        if any(
            abs(np.linalg.det(directions[list(ids)])) > 0.25
            for ids in combinations(range(len(directions)), 3)
        ):
            corners.append(hull.vertices[v].tolist())
    report = {
        "scale_m": scale,
        "patches": patches,
        "face_labels": labels.tolist(),
        "major_edges": edges,
        "rim_loops": loops,
        "corner_anchors_m": corners,
        "thresholds": {
            "plane_angle_deg": 10,
            "plane_distance_relative": 0.004,
            "edge_dihedral_deg": 28,
            "rim_dihedral_deg": 15,
            "rim_min_area_fraction": 0.035,
            "rim_min_sharp_boundary_fraction": 0.55,
            "merged_plane_distance_relative": 0.012,
            "merged_plane_angle_deg": 12,
            "rim_local_turning_band_relative": 0.065,
            "rim_local_turning_deg": 25,
        },
    }
    report["box"] = box_frame(hull, report, hint)
    return report


def box_frame(hull, structure, hint):
    if hint not in {"auto", "open_box", "generic"}:
        raise ValueError("shape_hint must be auto, open_box or generic")
    patches = structure["patches"]
    normals = np.array([p["normal"] for p in patches])
    areas = np.array([p["area_m2"] for p in patches])
    groups = []
    for index in np.argsort(-areas, kind="stable"):
        if areas[index] < 0.005 * hull.area:
            continue
        if not any(
            abs(normals[index] @ axis) > np.cos(np.deg2rad(14)) for axis in groups
        ):
            groups.append(normals[index])
    best = None
    for triple in combinations(groups[:18], 3):
        axes = np.array(triple).T
        if np.max(np.abs(axes.T @ axes - np.eye(3))) > 0.25:
            continue
        u, _, vt = np.linalg.svd(axes)
        axes = u @ vt
        alignment = np.abs(normals @ axes)
        matched = alignment.max(axis=1) >= np.cos(np.deg2rad(14))
        assignment = alignment.argmax(axis=1)
        explained = float(areas[matched].sum() / hull.area)
        # Both support-plane signs on all three axes, not just an OBB shape.
        stable = all(
            np.sum(
                areas[
                    matched
                    & (assignment == axis)
                    & ((normals @ axes[:, axis]) * sign > 0)
                ]
            )
            >= 0.008 * hull.area
            for axis in range(3)
            for sign in (-1, 1)
        )
        if stable and (best is None or explained > best[0]):
            best = (explained, axes)
    if best is None:
        return {
            "enabled": False,
            "confidence": 0.0,
            "reason": "No supported three-axis envelope",
            "hint": hint,
        }
    score, axes = best
    # Refine on ALL hull facets, not the order-dependent merged-patch seed.
    # Area-weighted signed normal families make the fitted frame reproducible
    # under rigid rotation and changes to coplanar hull triangulation.
    for _ in range(30):
        dots = hull.face_normals @ axes
        assignment = np.abs(dots).argmax(axis=1)
        aligned = np.abs(dots).max(axis=1) >= np.cos(np.deg2rad(14))
        refined = []
        for axis in range(3):
            ids = aligned & (assignment == axis)
            vector = np.sum(
                hull.face_normals[ids]
                * (hull.area_faces[ids] * np.sign(dots[ids, axis]))[:, None],
                axis=0,
            )
            refined.append(vector / np.linalg.norm(vector))
        u, _, vt = np.linalg.svd(np.array(refined).T)
        updated = u @ vt
        difference = np.linalg.norm(updated - axes)
        axes = updated
        if difference < 1e-12:
            break
    score = float(
        hull.area_faces[
            np.abs(hull.face_normals @ axes).max(axis=1) >= np.cos(np.deg2rad(14))
        ].sum()
        / hull.area
    )
    origin = hull.center_mass
    coordinates = (hull.vertices - origin) @ axes
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    threshold = 0.80 if hint == "auto" else 0.66
    return {
        "enabled": bool(hint != "generic" and score >= threshold),
        "confidence": score,
        "reason": "supported orthogonal plane families"
        if score >= threshold
        else "insufficient explained area",
        "hint": hint,
        "threshold": threshold,
        "axes_columns": axes.tolist(),
        "origin_m": origin.tolist(),
        "low_m": low.tolist(),
        "high_m": high.tolist(),
        "lengths_m": (high - low).tolist(),
    }
