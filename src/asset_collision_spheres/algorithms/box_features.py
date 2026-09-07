"""Conservative, mesh-supported planar features; no semantic or bounding-box corners.

Thresholds are dimensionless system defaults. Features guide sampling/selection;
they never replace the reference material mesh or establish material membership.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.material import _closest_point_exact


@dataclass
class BoxFeatures:
    planes: list
    segments: list
    anchors: np.ndarray
    face_labels: np.ndarray
    confidence: float
    enabled: bool
    reason: str
    scale: float

    def report(self):
        return {
            "feature_detection_confidence": self.confidence,
            "enabled": self.enabled,
            "reason": self.reason,
            "scale_m": self.scale,
            "planes": self.planes,
            "edge_segments": self.segments,
            "corner_anchors_m": self.anchors.tolist(),
            "corner_anchor_count": len(self.anchors),
            "edge_segment_count": len(self.segments),
            "definition": "mesh-supported box-like planes/interfaces, not a semantic open-box or cavity-topology proof",
        }


def area_length(mesh):
    """Surface-area-equivalent length; rigid/triangulation invariant, not max extent."""
    return float(np.sqrt(mesh.area / (2 * np.pi)))


def _fit_plane(points, areas):
    center = np.average(points, axis=0, weights=areas)
    delta = points - center
    values, vectors = np.linalg.eigh((delta * areas[:, None]).T @ delta / areas.sum())
    return center, vectors[:, 0], float(np.sqrt(max(values[0], 0)))


def detect_box_features(mesh, hint="auto"):
    if hint not in {"auto", "open_box", "generic"}:
        raise ValueError("shape_hint must be auto, open_box or generic")
    scale = area_length(mesh)
    labels = np.full(len(mesh.faces), -1, dtype=int)
    empty = BoxFeatures(
        [], [], np.empty((0, 3)), labels, 0.0, False, "generic requested", scale
    )
    if hint == "generic":
        return empty
    normals = mesh.face_normals
    centers = mesh.triangles_center
    areas = mesh.area_faces
    adjacent = mesh.face_adjacency
    graph = [[] for _ in areas]
    for a, b in adjacent:
        graph[a].append(int(b))
        graph[b].append(int(a))
    # Global seed normal and seed plane bounds prevent gradual drift around a bowl.
    # Signed normals and connected growth keep opposing thin inner/outer walls apart.
    available = np.ones(len(areas), dtype=bool)
    planes = []
    cos_angle = np.cos(np.deg2rad(16))
    for seed in np.argsort(-areas, kind="stable"):
        if not available[seed]:
            continue
        accepted = []
        queue = [int(seed)]
        seen = {int(seed)}
        while queue:
            face = queue.pop()
            if not available[face]:
                continue
            if normals[face] @ normals[seed] < cos_angle:
                continue
            if abs((centers[face] - centers[seed]) @ normals[seed]) > 0.007 * scale:
                continue
            accepted.append(face)
            available[face] = False
            for other in graph[face]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        ids = np.asarray(accepted, dtype=int)
        if not len(ids) or areas[ids].sum() < 0.008 * mesh.area:
            continue
        vertices = mesh.triangles[ids].reshape(-1, 3)
        center, normal, residual = _fit_plane(vertices, np.repeat(areas[ids] / 3, 3))
        if normal @ np.average(normals[ids], axis=0, weights=areas[ids]) < 0:
            normal = -normal
        if residual > 0.005 * scale:
            continue
        labels[ids] = len(planes)
        confidence = float(np.clip(1 - residual / (0.008 * scale), 0, 1))
        planes.append(
            {
                "id": len(planes),
                "center_m": center.tolist(),
                "normal": normal.tolist(),
                "area_m2": float(areas[ids].sum()),
                "residual_m": residual,
                "confidence": confidence,
                "face_ids": ids.tolist(),
                "adjacent_plane_ids": [],
                "support_vertices_m": vertices[:: max(1, len(vertices) // 80)].tolist(),
            }
        )
    if len(planes) < 3:
        return BoxFeatures(
            planes,
            [],
            empty.anchors,
            labels,
            0.0,
            False,
            "insufficient connected planar support",
            scale,
        )
    pn = np.asarray([p["normal"] for p in planes])
    pa = np.asarray([p["area_m2"] for p in planes])
    # Seek three approximately perpendicular supported directions. Axes alone
    # are not features: every segment below must also have an actual mesh bridge.
    best = 0.0
    for i in range(len(planes)):
        for j in range(i):
            if abs(pn[i] @ pn[j]) > 0.25:
                continue
            third = np.cross(pn[i], pn[j])
            third /= np.linalg.norm(third)
            axes = np.asarray([pn[i], pn[j], third])
            aligned = np.max(np.abs(pn @ axes.T), axis=1) > np.cos(np.deg2rad(13))
            counts = np.array([pa[np.abs(pn @ axis) > 0.96].sum() for axis in axes])
            if counts.min() < 0.015 * mesh.area:
                continue
            best = max(best, float(pa[aligned].sum() / mesh.area))
    enabled = best >= (0.48 if hint == "open_box" else 0.58)
    if not enabled:
        return BoxFeatures(
            planes,
            [],
            empty.anchors,
            labels,
            best,
            False,
            "planar directions insufficiently box-like; generic fallback",
            scale,
        )

    # Multi-source propagation only over short material-surface bridges. This
    # admits bevels/rims while never joining disconnected planes across air.
    owners = labels.copy()
    distances = np.where(labels >= 0, 0.0, np.inf)
    # Minimum face altitude measures traversal through a narrow rim/bevel rather
    # than charging its long triangle centroid distance (bad on coarse meshes).
    edge_lengths = np.linalg.norm(
        mesh.triangles - np.roll(mesh.triangles, 1, axis=1), axis=2
    )
    width = 2 * areas / edge_lengths.max(axis=1)
    heap = []
    for face in np.flatnonzero(labels >= 0):
        for other in graph[face]:
            if labels[other] < 0:
                heapq.heappush(
                    heap, (float(width[other] * 0.5), int(other), int(labels[face]))
                )
    while heap:
        distance, face, owner = heapq.heappop(heap)
        if distance >= distances[face] or distance > scale * 0.045:
            continue
        distances[face], owners[face] = distance, owner
        for other in graph[face]:
            candidate = distance + (width[face] + width[other]) * 0.5
            if candidate < distances[other]:
                heapq.heappush(heap, (float(candidate), other, owner))
    # Rejoin noisy fragments only when a short *material graph* bridge connects
    # them and signed normals/plane offsets agree. Never join parallel layers
    # by orientation alone (in particular, opposing inner/outer wall normals).
    parent = list(range(len(planes)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for pair in adjacent:
        a, b = owners[pair]
        if a < 0 or b < 0 or a == b or pn[a] @ pn[b] < 0.99:
            continue
        delta = np.asarray(planes[a]["center_m"]) - planes[b]["center_m"]
        if max(abs(delta @ pn[a]), abs(delta @ pn[b])) < scale * 0.009:
            parent[root(b)] = root(a)
    groups = {}
    for i in range(len(planes)):
        groups.setdefault(root(i), []).append(i)
    validated_groups = []
    for indices in groups.values():
        if len(indices) > 1:
            ids = np.concatenate([planes[i]["face_ids"] for i in indices]).astype(int)
            _, normal, residual = _fit_plane(
                mesh.triangles[ids].reshape(-1, 3), np.repeat(areas[ids] / 3, 3)
            )
            if normal @ pn[indices[0]] < 0:
                normal = -normal
            if residual > 0.005 * scale or np.min(pn[indices] @ normal) < cos_angle:
                # Transitive pairwise matches must not gradually merge a curve.
                validated_groups.extend([[i] for i in indices])
                continue
        validated_groups.append(indices)
    remap = np.empty(len(planes), dtype=int)
    merged = []
    for indices in validated_groups:
        ids = np.concatenate([planes[i]["face_ids"] for i in indices]).astype(int)
        vertices = mesh.triangles[ids].reshape(-1, 3)
        center, normal, residual = _fit_plane(vertices, np.repeat(areas[ids] / 3, 3))
        if normal @ pn[indices[0]] < 0:
            normal = -normal
        # Keep the merged fit auditable; no geometry or material-query changes.
        p = dict(planes[indices[0]])
        p.update(
            id=len(merged),
            center_m=center.tolist(),
            normal=normal.tolist(),
            residual_m=residual,
            area_m2=float(areas[ids].sum()),
            face_ids=ids.tolist(),
            confidence=float(np.clip(1 - residual / (0.008 * scale), 0, 1)),
            support_vertices_m=vertices[:: max(1, len(vertices) // 80)].tolist(),
        )
        remap[indices] = len(merged)
        merged.append(p)
    labels[labels >= 0] = remap[labels[labels >= 0]]
    owners[owners >= 0] = remap[owners[owners >= 0]]
    planes = merged
    pn = np.asarray([p["normal"] for p in planes])
    interfaces = {}
    for pair, edge in zip(adjacent, mesh.face_adjacency_edges):
        a, b = owners[pair]
        if a < 0 or b < 0 or a == b:
            continue
        angle = float(pn[a] @ pn[b])
        if abs(angle) > 0.30 and angle > -0.94:
            continue
        key = tuple(sorted((int(a), int(b))))
        interfaces.setdefault(key, []).append(edge)
    segments = []
    for (a, b), edges in interfaces.items():
        planes[a]["adjacent_plane_ids"].append(b)
        planes[b]["adjacent_plane_ids"].append(a)
        # Split by actual interface vertex connectivity, so holes and gaps are
        # not bridged merely by two infinite fitted planes intersecting.
        neighbours = {}
        for u, v in edges:
            neighbours.setdefault(int(u), set()).add(int(v))
            neighbours.setdefault(int(v), set()).add(int(u))
        remaining = set(neighbours)
        while remaining:
            first = min(remaining)
            group, todo = set(), [first]
            while todo:
                node = todo.pop()
                if node in group:
                    continue
                group.add(node)
                todo.extend(neighbours[node] - group)
            remaining -= group
            points = mesh.vertices[sorted(group)]
            if len(points) < 2:
                continue
            origin = points.mean(axis=0)
            _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
            direction = basis[0]
            projected = (points - origin) @ direction
            residual = np.linalg.norm(
                points - origin - projected[:, None] * direction, axis=1
            )
            length = float(np.ptp(projected))
            if length < 0.09 * scale or np.quantile(residual, 0.9) > 0.025 * scale:
                continue
            # Endpoints are actual supporting mesh vertices, not extrapolations.
            ends = points[[int(np.argmin(projected)), int(np.argmax(projected))]]
            segments.append(
                {
                    "id": len(segments),
                    "plane_ids": [a, b],
                    "endpoints_m": ends.tolist(),
                    "length_m": float(np.linalg.norm(ends[1] - ends[0])),
                    "confidence": float(
                        min(planes[a]["confidence"], planes[b]["confidence"])
                    ),
                    "type": "rim_bridge" if pn[a] @ pn[b] < -0.94 else "crease_bridge",
                    "support_vertex_ids": sorted(group),
                }
            )
    # Corners need two non-parallel *supported* segments meeting locally.
    anchors = []
    ends = np.asarray([p for s in segments for p in s["endpoints_m"]]).reshape(-1, 3)
    if len(ends):
        tree = cKDTree(ends)
        suppressed = set()
        for i, point in enumerate(ends):
            if i in suppressed:
                continue
            nearby = tree.query_ball_point(point, scale * 0.05)
            directions = []
            for index in nearby:
                e = np.asarray(segments[index // 2]["endpoints_m"])
                directions.append((e[1] - e[0]) / np.linalg.norm(e[1] - e[0]))
            if (
                len(directions) < 2
                or np.min(np.abs(np.asarray(directions) @ np.asarray(directions).T))
                > 0.8
            ):
                continue
            location, _, _ = _closest_point_exact(
                mesh, np.asarray([np.mean(ends[nearby], axis=0)])
            )
            anchors.append(location[0])
            suppressed.update(nearby)
    if not segments:
        enabled = False
    return BoxFeatures(
        planes,
        segments,
        np.asarray(anchors).reshape(-1, 3),
        labels,
        best,
        enabled,
        "mesh-supported box-like features"
        if enabled
        else "no reliable material interfaces; generic fallback",
        scale,
    )


def feature_samples(mesh, features, per_edge=13):
    """Project segment witnesses onto the original material, never into a hole."""
    points, ids = [], []
    for segment in features.segments if features.enabled else []:
        a, b = np.asarray(segment["endpoints_m"])
        p = a + np.linspace(0, 1, per_edge)[:, None] * (b - a)
        nearest, distance, _ = _closest_point_exact(mesh, p)
        keep = distance < 0.025 * features.scale
        points.extend(nearest[keep])
        ids.extend([segment["id"]] * int(keep.sum()))
    return np.asarray(points).reshape(-1, 3), np.asarray(ids, dtype=int)


def corner_witnesses(mesh, features):
    """A corner is a material neighbourhood, not a point hit by a microscopic ball.

    Use the anchor and two nonparallel, short material-supported directions.
    These are representation targets; they impose no new radius/material rule.
    """
    witnesses = []
    for anchor in features.anchors:
        directions = []
        for segment in features.segments:
            a, b = np.asarray(segment["endpoints_m"])
            if np.linalg.norm(b - anchor) < np.linalg.norm(a - anchor):
                a, b = b, a
            if np.linalg.norm(a - anchor) > features.scale * 0.06:
                continue
            direction = (b - a) / np.linalg.norm(b - a)
            if not directions or abs(direction @ directions[0]) < 0.8:
                directions.append(direction)
            if len(directions) == 2:
                break
        if len(directions) != 2:
            raise ValueError(
                "Detected corner lost its two supported incident directions"
            )
        proposed = np.r_[
            anchor[None], anchor + np.asarray(directions) * (0.015 * features.scale)
        ]
        nearest, _, _ = _closest_point_exact(mesh, proposed)
        witnesses.append(nearest)
    return np.asarray(witnesses).reshape(-1, 3, 3)


def evaluate_features(mesh, spheres, features):
    from ..geometry.material import sphere_clearance

    report = features.report()
    tolerance = 0.006 * features.scale
    anchors = features.anchors if features.enabled else np.empty((0, 3))
    neighbourhoods = (
        corner_witnesses(mesh, features) if len(anchors) else np.empty((0, 3, 3))
    )
    support = (
        (
            sphere_clearance(neighbourhoods.reshape(-1, 3), spheres)
            <= 0.001 * features.scale
        )
        .reshape(-1, 3)
        .all(axis=1)
    )
    points, ids = feature_samples(
        mesh, features, per_edge=41
    )  # held-out denser evaluation
    covered = (
        sphere_clearance(points, spheres) <= tolerance
        if len(points)
        else np.empty(0, bool)
    )
    gaps, scores, empty_edges, empty_middles = [], [], [], []
    for segment in features.segments if features.enabled else []:
        hits = covered[ids == segment["id"]]
        longest = current = 0
        for hit in hits:
            current = 0 if hit else current + 1
            longest = max(longest, current)
        gaps.append(segment["length_m"] * min(1, longest / max(len(hits) - 1, 1)))
        scores.append(
            float(np.mean([np.any(part) for part in np.array_split(hits, 3)]))
        )
        if not hits.any():
            empty_edges.append(segment["id"])
        if not np.array_split(hits, 3)[1].any():
            empty_middles.append(segment["id"])
    major_length = max((s["length_m"] for s in features.segments), default=0) * 0.5
    missing_major = [
        s["id"]
        for s in features.segments
        if s["id"] in empty_edges and s["length_m"] >= major_length
    ]
    report.update(
        corner_anchor_supported_count=int(support.sum()),
        unsupported_corner_ids=np.flatnonzero(~support).tolist(),
        corner_neighbourhood_witnesses_m=neighbourhoods.tolist(),
        edge_max_gap_m=max(gaps, default=0.0),
        edge_gaps_m=gaps,
        edge_distribution_score=float(np.mean(scores)) if scores else None,
        unrepresented_edge_ids=empty_edges,
        unrepresented_middle_third_ids=empty_middles,
        major_edges_without_representation=missing_major,
        representation_neighbourhood_m=tolerance,
        evaluation_note="corner=3 material witnesses spanning 1.5% area-length; edge=41 held-out witnesses; finite sparse representation, not continuous collision coverage",
    )
    return report
