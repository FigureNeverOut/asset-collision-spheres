"""Precision-first container-envelope eligibility, NOT hollow/solid semantics.

A candidate unsupported hull patch must lead to persistent free space enclosed
by material in several actual sections, and terminate at a bottom. All gates
are shared dimensionless geometry policy, never asset-specific annotations.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .hull_structure import analyze, area_samples, convex_hull, project

THRESHOLDS = {
    "unsupported_distance_scale": 0.025,
    "cap_min_area_fraction": 0.04,
    "cap_min_unsupported_fraction": 0.55,
    "cap_min_added_area_concentration": 0.30,
    "free_depth_min_extent": 0.12,
    "bottom_hit_fraction_min": 0.65,
    "section_depth_fractions": [0.2, 0.4, 0.6],
    "section_radial_directions": 24,
    "section_min_enclosure": 0.95,
    "section_min_consistency": 1.0,
    "elongation_limit": 8.0,
    "confidence_min": 0.78,
}


def first_hit(mesh, origins, direction):
    origins = np.asarray(origins).reshape(-1, 3)
    directions = np.broadcast_to(direction, origins.shape).copy()
    locations, rays, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=True
    )
    distances = np.full(len(origins), np.inf)
    # Trimesh versions may return shape (0,) instead of (0, 3) on no hits.
    if len(rays) == 0:
        return distances
    along = np.einsum("ij,ij->i", locations - origins[rays], directions[rays])
    good = along > 1e-9
    np.minimum.at(distances, rays[good], along[good])
    return distances


def section_evidence(mesh, center, normal, depth, scale):
    axes = np.eye(3)
    u = np.cross(normal, axes[np.argmin(np.abs(normal))])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    result = []
    for fraction in THRESHOLDS["section_depth_fractions"]:
        origin = center - normal * (depth * fraction)
        segments = trimesh.intersections.mesh_plane(mesh, normal, origin)
        crossings = []
        if len(segments):
            xy = (segments - origin) @ np.column_stack((u, v))
            p, s = xy[:, 0], xy[:, 1] - xy[:, 0]
            for angle in np.arange(24) * 2 * np.pi / 24:
                d = np.array([np.cos(angle), np.sin(angle)])
                cross = d[0] * s[:, 1] - d[1] * s[:, 0]
                valid = np.abs(cross) > scale * 1e-12
                t = np.divide(
                    p[:, 0] * s[:, 1] - p[:, 1] * s[:, 0],
                    cross,
                    out=np.zeros(len(s)),
                    where=valid,
                )
                along_segment = np.divide(
                    p[:, 0] * d[1] - p[:, 1] * d[0],
                    cross,
                    out=np.zeros(len(s)),
                    where=valid,
                )
                hit = np.sort(
                    t[
                        valid
                        & (t > scale * 1e-7)
                        & (along_segment >= -1e-9)
                        & (along_segment <= 1 + 1e-9)
                    ]
                )
                unique = (
                    hit[np.r_[True, np.diff(hit) > scale * 1e-6]] if len(hit) else hit
                )
                crossings.append(len(unique))
        enclosure = float(np.mean(np.asarray(crossings) >= 2)) if crossings else 0.0
        # A single 3D parity ray proved unstable on the real scan's thin wall.
        # Consensus of 24 independent IN-PLANE rays explicitly distinguishes a
        # free interior (even crossings) from solid material (odd crossings).
        parity = float(np.mean(np.asarray(crossings) % 2 == 0)) if crossings else 0.0
        free = parity >= THRESHOLDS["section_min_enclosure"]
        result.append(
            {
                "depth_fraction": fraction,
                "origin_m": origin.tolist(),
                "origin_in_free_space": free,
                "material_enclosure_fraction": enclosure,
                "free_space_even_parity_fraction": parity,
                "radial_crossing_counts": crossings,
                "segment_count": len(segments),
                "passed": free and enclosure >= THRESHOLDS["section_min_enclosure"],
            }
        )
    return result


def route_geometry(mesh):
    from .original_surface import surface_mesh

    mesh = surface_mesh(mesh)
    decision = {
        "geometry_policy_requested": "auto",
        "geometry_policy_selected": "original_surface",
        "container_envelope_confidence": 0.0,
        "review_required": False,
        "reason": "No strong container-envelope evidence; preserve original surface",
        "fallback_reason": None,
        "thresholds": THRESHOLDS,
        "evidence": {},
        "candidate_caps": [],
    }
    try:
        hull = convex_hull(mesh)
    except ValueError as error:
        decision.update(
            review_required=True,
            reason="Analysis hull unavailable; preserve original surface",
            fallback_reason=str(error),
        )
        return decision
    structure = analyze(hull)
    scale = structure["scale_m"]
    points, face_ids = area_samples(hull, 4096, 91827)
    _, distance, _ = project(mesh, points)
    unsupported = distance > scale * THRESHOLDS["unsupported_distance_scale"]
    labels = np.asarray(structure["face_labels"])[face_ids]
    # Use area-sampled covariance, not tessellation-dependent vertex PCA.
    eigen = np.linalg.eigvalsh(np.cov(points.T))
    elongation = float(np.sqrt(eigen[-1] / max(eigen[-2], 1e-30)))
    trust = bool(mesh.is_volume)
    fill = float(mesh.volume / hull.volume) if trust else None
    decision["evidence"].update(
        fill_ratio=fill,
        fill_ratio_trustworthy=trust,
        unsupported_hull_area_fraction=float(unsupported.mean()),
        elongation=elongation,
        elongation_or_branching_penalty=float(np.clip((elongation - 4) / 4, 0, 1)),
        branching_evidence_scope="Only elongated-envelope and multi-patch counterevidence; no skeleton classifier",
        analysis_hull_volume_m3=float(hull.volume),
        analysis_hull_area_m2=float(hull.area),
    )
    decision["analysis_hull_vertices_m"] = hull.vertices.tolist()
    decision["analysis_hull_faces"] = hull.faces.tolist()
    candidates = []
    for patch in structure["patches"]:
        mask = labels == patch["id"]
        added = mask & unsupported
        fraction = float(added.sum() / max(mask.sum(), 1))
        area_fraction = patch["area_m2"] / hull.area
        if area_fraction < 0.04 or fraction < 0.55:
            continue
        candidates.append((patch, added, fraction, area_fraction))
    decision["evidence"]["cap_patch_count"] = len(candidates)
    decision["evidence"]["cap_area_fraction"] = float(sum(c[3] for c in candidates))
    if not trust:
        decision.update(
            review_required=True,
            reason="Material membership unreliable; no automatic hull approval",
            fallback_reason="Open, inconsistent or non-positive-volume source mesh",
        )
        return decision
    for patch, added, fraction, area_fraction in sorted(
        candidates, key=lambda x: -int(x[1].sum())
    )[:4]:
        normal = np.asarray(patch["normal"])
        cap_points = points[added]
        center = cap_points.mean(axis=0)
        center -= normal * np.dot(center - np.asarray(patch["center_m"]), normal)
        extent = float(np.ptp(hull.vertices @ normal))
        epsilon = max(scale * 1e-5, 1e-9)
        origins = (
            cap_points[
                np.linspace(0, len(cap_points) - 1, min(25, len(cap_points)), dtype=int)
            ]
            - normal * epsilon
        )
        depths = first_hit(mesh, origins, -normal)
        bottom_fraction = float(
            np.mean(
                np.isfinite(depths)
                & (depths >= extent * 0.12)
                & (depths <= extent * 1.01)
            )
        )
        center_depth = float(first_hit(mesh, [center - normal * epsilon], -normal)[0])
        free_depth = bool(
            np.isfinite(center_depth) and 0.12 * extent <= center_depth <= 1.01 * extent
        )
        sections = (
            section_evidence(mesh, center, normal, center_depth, scale)
            if free_depth
            else []
        )
        consistency = (
            float(np.mean([s["passed"] for s in sections])) if sections else 0.0
        )
        concentration = float(added.sum() / max(unsupported.sum(), 1))
        # Fill ratio is only 5% of the score, never the approval gate.
        confidence = float(
            0.25 * min(concentration / 0.6, 1)
            + 0.20 * fraction
            + 0.25 * consistency
            + 0.25 * bottom_fraction
            + 0.05 * np.clip(1 - fill, 0, 1)
        )
        gates = {
            "concentrated_cap": concentration >= 0.30,
            "persistent_free_depth": free_depth,
            "bottom_evidence": bottom_fraction >= 0.65,
            "closed_material_sections": consistency >= 1.0,
            "not_extremely_elongated": elongation < 8.0,
            "confidence": confidence >= 0.78,
        }
        candidate = {
            "patch_id": patch["id"],
            "center_m": center.tolist(),
            "normal": normal.tolist(),
            "area_fraction": float(area_fraction),
            "unsupported_fraction": fraction,
            "added_area_concentration": concentration,
            "bottom_hit_fraction": bottom_fraction,
            "center_free_depth_m": center_depth if np.isfinite(center_depth) else None,
            "axis_extent_m": extent,
            "sections": sections,
            "section_consistency": consistency,
            "confidence": confidence,
            "gates": gates,
            "eligible": all(gates.values()),
        }
        decision["candidate_caps"].append(candidate)
    best = max(decision["candidate_caps"], key=lambda c: c["confidence"], default=None)
    eligible = [c for c in decision["candidate_caps"] if c["eligible"]]
    if eligible:
        best = max(eligible, key=lambda c: c["confidence"])
        decision.update(
            geometry_policy_selected="convex_hull",
            reason="Concentrated unsupported cap, persistent free depth, enclosing sections and bottom agree",
        )
    elif candidates or unsupported.mean() > 0.1:
        decision.update(
            review_required=True,
            fallback_reason="Candidate cap failed combined eligibility gates"
            if candidates
            else "Significant unsupported hull area but no qualifying cap",
        )
    if best:
        decision["container_envelope_confidence"] = best["confidence"]
        decision["evidence"].update(
            selected_candidate_patch_id=best["patch_id"],
            section_consistency=best["section_consistency"],
            cap_added_area_concentration=best["added_area_concentration"],
            failed_gates=[k for k, v in best["gates"].items() if not v],
        )
    return decision
