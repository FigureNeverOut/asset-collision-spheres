"""Automatic local region profile for the flat solid-box family.

This experimental family adapter derives regions from geometry, not asset names
or hand-authored book coordinates. It reuses the unchanged full-mesh generator.
Thick blocks and non-axis-aligned/non-box meshes are explicitly out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_VERSION = "solid-box-flat-regions-v1"


@dataclass(frozen=True)
class SolidBoxRegionProfile:
    config: dict[str, Any]
    facts: dict[str, Any]


def _quotas(weights, budget):
    ideal = np.asarray(weights) * budget
    counts = np.maximum(1, np.floor(ideal).astype(int))
    while counts.sum() > budget:
        excess = np.where(counts > 1, counts - ideal, -np.inf)
        counts[int(np.argmax(excess))] -= 1
    while counts.sum() < budget:
        counts[int(np.argmax(ideal - counts))] += 1
    return counts.tolist()


def make_solid_box_region_profile(mesh, *, budget=64, max_outward_offset_m=0.006, probe_radius_m=0.006, seed=5):
    """Nine coverage regions across the two broad axes; no geometry replacement."""
    if type(budget) is not int or not 9 <= budget <= 256:
        raise ValueError("Flat solid-box profile requires an exact budget in [9, 256]")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Flat solid-box profile requires a closed, consistently oriented solid")
    bounds = np.asarray(mesh.bounds)
    extents = bounds[1] - bounds[0]
    if not np.isfinite(bounds).all() or np.any(extents <= 0):
        raise ValueError("Invalid solid-box bounds")
    order = np.argsort(extents)
    thin, u, v = int(order[0]), int(order[2]), int(order[1])
    thin_ratio = float(extents[thin] / extents[v])
    hull_ratio = float(mesh.volume / mesh.convex_hull.volume)
    box_ratio = float(mesh.volume / np.prod(extents))
    if thin_ratio > 0.25:
        raise ValueError("Thick solid boxes need a separate profile; the current family adapter is flat-only")
    if hull_ratio < 0.98 or box_ratio < 0.85:
        raise ValueError(
            "Mesh is not sufficiently convex and box-like in its local frame; no automatic box substitution"
        )
    if isinstance(max_outward_offset_m, bool) or not np.isfinite(max_outward_offset_m) or max_outward_offset_m < 0:
        raise ValueError("max_outward_offset_m must be finite and nonnegative")
    if isinstance(probe_radius_m, bool) or not np.isfinite(probe_radius_m) or probe_radius_m <= 0:
        raise ValueError("probe_radius_m must be finite and positive")
    cuts = np.array([0.0, 0.25, 0.75, 1.0])
    weights = [float((cuts[i + 1] - cuts[i]) * (cuts[j + 1] - cuts[j])) for i in range(3) for j in range(3)]
    counts = _quotas(weights, budget)
    radius_cap = float(extents[thin] / 2 + max_outward_offset_m)
    regions = []
    for i in range(3):
        for j in range(3):
            group = "center" if i == j == 1 else "corner" if i != 1 and j != 1 else "edge"
            local = bounds.copy()
            local[:, u] = bounds[0, u] + extents[u] * cuts[i : i + 2]
            local[:, v] = bounds[0, v] + extents[v] * cuts[j : j + 2]
            regions.append(
                {
                    "name": f"{group}_{i}_{j}",
                    "bounds_m": local.tolist(),
                    "sphere_count": counts[3 * i + j],
                    "max_radius_m": radius_cap,
                }
            )
    free_paths = []
    for sign, name in ((-1, "lower"), (1, "upper")):
        start = bounds.mean(0)
        start[thin] = bounds[0 if sign < 0 else 1, thin] + sign * (max_outward_offset_m + probe_radius_m + 0.001)
        end = start.copy()
        start[u], end[u] = bounds[0, u] - 0.03, bounds[1, u] + 0.03
        free_paths.append(
            {
                "name": name + "_broad_face_clearance",
                "start_m": start.tolist(),
                "end_m": end.tolist(),
                "radius_m": probe_radius_m,
                "count": 101,
            }
        )
    return SolidBoxRegionProfile(
        config={
            "description": (
                "Auto-derived flat solid-box profile; nine broad-plane regions, complete actual collision mesh"
            ),
            "sphere_budget": budget,
            "max_outward_offset_m": max_outward_offset_m,
            "expected_bounds_m": bounds.tolist(),
            "bounds_tolerance_m": 2e-6,
            "candidate_sample_count": 6000,
            "seed": seed,
            "probe_radius_m": probe_radius_m,
            "probe_penetration_m": min(0.0015, probe_radius_m * 0.25),
            "regions": regions,
            "free_probe_paths": free_paths,
        },
        facts={
            "profile_version": PROFILE_VERSION,
            "family": "solid_box",
            "subtype": "flat",
            "thin_axis_index": thin,
            "broad_axis_indices": [u, v],
            "extents_m": extents.tolist(),
            "thin_to_second_axis_ratio": thin_ratio,
            "material_to_convex_hull_volume_ratio": hull_ratio,
            "material_to_aabb_volume_ratio": box_ratio,
            "region_counts": counts,
            "geometry_replaced": False,
            "production_adapter_enabled": False,
            "scope": (
                "Explicit local-axis flat solid boxes; dimension ratios are admission checks, "
                "not automatic semantic classification"
            ),
        },
    )
