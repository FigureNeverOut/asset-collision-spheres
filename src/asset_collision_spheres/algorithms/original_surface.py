"""Original triangle surface sites, never projected onto the analysis hull."""

from __future__ import annotations

import time

import numpy as np
import trimesh

from ..geometry.types import MeshRegionSphereResult
from ..geometry.validation import mesh_fingerprint
from .convex_surface import POLICY, box_grid, shared_radius, stats
from .hull_diagnostics import evaluate
from .hull_structure import analyze, area_samples, convex_hull, project

VERSION = "original-surface-v4-experimental"


def original_structure(target, hint):
    structure = analyze(target, "auto" if hint == "box" else hint)
    # Orthogonal face normals alone would label U/L pieces as boxes. Require
    # the ORIGINAL surface to nearly match its envelope, never fill a concavity.
    if structure["box"]["enabled"]:
        try:
            hull = convex_hull(target)
            sample, _ = area_samples(target, 1024, 771)
            _, distance, _ = project(hull, sample)
            volume_ratio = (
                float(target.volume / hull.volume) if target.is_volume else None
            )
            area_ratio = float(target.area / hull.area)
            near_envelope = bool(
                np.quantile(distance, 0.95) <= 0.025 * structure["scale_m"]
                and distance.max() <= 0.08 * structure["scale_m"]
            )
            eligible = (volume_ratio is not None and volume_ratio >= 0.85) or (
                volume_ratio is None
                and 0.85 <= area_ratio <= 1.10
                and np.quantile(distance, 0.95) <= 0.015 * structure["scale_m"]
            )
            eligible = eligible and near_envelope
            structure["box"]["original_envelope_check"] = {
                "volume_ratio": volume_ratio,
                "area_ratio": area_ratio,
                "surface_to_hull_p95_m": float(np.quantile(distance, 0.95)),
                "surface_to_hull_max_m": float(distance.max()),
                "passed": bool(eligible),
            }
            if not eligible:
                structure["box"].update(
                    enabled=False,
                    reason="Original concavity must not be replaced by box envelope",
                )
        except ValueError as error:
            structure["box"].update(enabled=False, reason=str(error))
    return structure


def surface_mesh(mesh):
    vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Original surface needs finite [V,3] vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Original surface needs actual triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Original surface triangle index out of range")
    # Weld only EXACT duplicate coordinates (export seams); never nearby arms.
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    target = trimesh.Trimesh(unique, inverse[faces], process=False)
    target.update_faces(target.area_faces > max(target.area, 1e-30) * 1e-14)
    target.remove_unreferenced_vertices()
    if not len(target.faces) or target.area < 1e-18:
        raise ValueError("Original surface has no usable area")
    return target


def generate_original_surface_spheres(mesh, config):
    started = time.perf_counter()
    target = surface_mesh(mesh)
    structure = original_structure(target, config["shape_hint"])
    sites, kinds = [], []
    count = config["effective_max_spheres"]
    selection = {}
    strategy = "original_graph_fps"
    if structure["box"]["enabled"] and count >= 8:
        sites, kinds, selection = box_grid(target, structure, count)
        count = selection["unique_boundary_target_count"]
        strategy = "original_box_grid"
    from .surface_graph import graph_fps

    if len(sites) < count:
        sites, kinds, connectivity, curve = graph_fps(
            target, count, config["seed"], sites, kinds
        )
    else:
        connectivity, curve = (
            {"method": "regular_box_grid; graph refinement not needed"},
            [],
        )
    spheres = np.column_stack((sites, np.full(len(sites), shared_radius(target))))
    _, errors, _ = project(target, spheres[:, :3])
    if errors.max() > max(1e-9, np.sqrt(target.area) * 1e-7):
        raise ValueError("Sites must lie on the original surface")
    diagnostics = evaluate(target, spheres, structure)
    diagnostics.update(
        algorithm_version=VERSION,
        mode="original_surface",
        target_geometry="original_surface",
        config=config,
        actual_count=len(spheres),
        effective_max_spheres=config["effective_max_spheres"],
        sampling_strategy=strategy,
        selection=selection,
        radius_m=stats(spheres[:, 3]),
        connectivity_fairness=connectivity,
        count_stop_reason="regular_boundary_grid"
        if strategy == "original_box_grid"
        else "effective_capacity_or_graph_exhaustion",
        radius_policy="L=sqrt(original.area/(2*pi)); r=min(0.10*L, 0.006+0.02*L, 0.012) metres",
        system_policy=POLICY,
        surface_site_max_error_m=float(errors.max()),
        surface_structure=structure,
        sphere_sources=kinds,
        original_mesh_sha256=mesh_fingerprint(mesh),
        target_area_m2=float(target.area),
        target_volume_m3=float(target.volume) if target.is_volume else None,
        incremental_reference_distance_m=curve,
        review_required=bool(
            selection.get("review_required")
            or connectivity.get("missing_components")
            or not target.is_watertight
            or (structure["box"]["enabled"] and count < 8)
        ),
        validation_status="finite_original_surface_checks_passed",
        acceptance="Sparse original surface approximation, not collision completeness",
        generation_time_s=time.perf_counter() - started,
    )
    return MeshRegionSphereResult(spheres, tuple(kinds), diagnostics)
