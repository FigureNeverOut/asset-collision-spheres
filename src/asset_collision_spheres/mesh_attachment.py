"""Full-material mesh adapter. No automatic convex fallback or semantic guesses."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from .mesh_region_spheres import MeshRegionSphereResult, material_depth, sphere_clearance

ADAPTER_VERSION = "full-material-native-v1"


def characteristic_length(mesh):
    """Rotation/translation-invariant size, also stable under planar subdivision."""
    return float(2 * np.linalg.norm(mesh.vertices - mesh.centroid, axis=1).max())


def sphere_bound_metrics(mesh, spheres, depths):
    expansion = np.maximum(np.asarray(spheres)[:, 3] - depths, 0)
    return {
        "protrusion": float(np.mean(expansion > 1e-8)),
        "protrusion_mean_m": float(expansion.mean()),
        "protrusion_p95_m": float(np.quantile(expansion, 0.95)),
        "volume_ratio": float(np.sum(4 * np.pi / 3 * np.asarray(spheres)[:, 3] ** 3) / mesh.volume),
        "metric_definitions": (
            "protrusion fields describe per-ball expansion bounds, NOT sampled outside volume; "
            "volume_ratio sums ball volumes without overlap correction"
        ),
    }


def mesh_fingerprint(mesh):
    """Object-local identity; tolerate sub-micrometre capture/export rounding only."""
    digest = hashlib.sha256(b"full-material-local-mesh-v1")
    digest.update(np.round(np.asarray(mesh.vertices) / 1e-6).astype("<i8").tobytes())
    digest.update(np.asarray(mesh.faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def check_material_mesh(mesh):
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Material mesh is empty or non-finite")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Material sign is uncertain: require closed, outward-oriented mesh; no hull fallback")
    if np.any(mesh.area_faces <= np.finfo(float).eps * np.linalg.norm(mesh.extents) ** 2):
        raise ValueError("Degenerate material triangles are unsupported")


def check_spheres(mesh, spheres, budget, expansion):
    check_material_mesh(mesh)
    spheres = np.asarray(spheres, dtype=float)
    if spheres.ndim != 2 or spheres.shape[1] != 4 or not 1 <= len(spheres) <= budget:
        raise ValueError("Sphere set must be nonempty Nx4 and within actual native capacity")
    if not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0):
        raise ValueError("Sphere coordinates/radii must be finite and radii positive")
    if isinstance(expansion, bool) or not np.isfinite(expansion) or expansion < 0:
        raise ValueError("max_outward_offset_m must be explicit, finite and nonnegative")
    depths = material_depth(mesh, spheres[:, :3], exact_distance=True)
    # Same 1 µm absolute radius tolerance as native slot readback. In particular
    # do not edit the approved legacy set for a 0.27 µm nearest-query correction.
    tolerance = max(1e-6, characteristic_length(mesh) * 1e-6)
    if np.any(depths <= 0) or np.any(spheres[:, 3] - depths > expansion + tolerance):
        raise ValueError("Invalid material centres or expansion bound; refuse native attachment")
    return depths


def prepare_request(raw, *, capacity, seed):
    """Resolve immutable artifact content before cache lookup (replacement invalidates)."""
    config = dict(raw)
    config.setdefault("mode", "auto_geometry")
    config.setdefault("sphere_budget", capacity)
    config.setdefault("seed", seed)
    if type(config["sphere_budget"]) is not int or not 1 <= config["sphere_budget"] <= capacity:
        raise ValueError("mesh sphere_budget must fit the reserved native capacity")
    if config["mode"] == "precomputed":
        if set(config) - {"mode", "sphere_budget", "seed", "artifact_path", "artifact_sha256"}:
            raise ValueError("Unknown precomputed mesh config options")
        path = Path(config["artifact_path"])
        if not path.is_absolute():
            raise ValueError("precomputed artifact_path must be absolute")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if config.get("artifact_sha256") != digest:
            raise ValueError("Precomputed artifact SHA256 mismatch")
        config["artifact"] = json.loads(content)
    return config, hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def fit_mesh_attachment(mesh, config):
    started = time.perf_counter()
    if config["mode"] != "precomputed":
        from .auto_geometry_spheres import generate_auto_geometry_spheres

        result = generate_auto_geometry_spheres(mesh, config)
        if result.diagnostics["missing_material_components"]:
            raise ValueError(
                "Entire material components have no sphere representation; inspect the diagnostic output "
                "and budget before native attachment. Ordinary contact misses are not this gate."
            )
        return result
    artifact = config["artifact"]
    if artifact.get("schema") != "fastsim/mesh-spheres/1" or artifact.get("mesh_sha256") != mesh_fingerprint(mesh):
        raise ValueError("Precomputed spheres do not match the captured object-local full mesh")
    spheres = np.asarray(artifact["spheres_object_local_m"], dtype=float)
    expansion = artifact["max_outward_offset_m"]
    depths = check_spheres(mesh, spheres, config["sphere_budget"], expansion)
    # These finite samples report shape quality, not a zero-miss acceptance gate.
    surface, _ = trimesh.sample.sample_surface(mesh, 2000, seed=config["seed"] + 101)
    gap = np.maximum(sphere_clearance(surface, spheres), 0)
    diagnostics = {
        "algorithm_version": ADAPTER_VERSION,
        "mode": "precomputed",
        "actual_count": len(spheres),
        "generation_time_s": time.perf_counter() - started,
        "max_outward_offset_m": expansion,
        "max_radius_minus_material_depth_m": float(np.max(spheres[:, 3] - depths)),
        "numerical_tolerance_m": max(1e-6, characteristic_length(mesh) * 1e-6),
        "validation_status": "finite_checks_passed",
        "surface_coverage": float(np.mean(gap <= 1e-8)),
        "surface_gap_mean_m": float(gap.mean()),
        "surface_gap_p95_m": float(np.quantile(gap, 0.95)),
        "max_uncovered_gap_m": float(gap.max()),
        "ordinary_sample_count": len(gap),
        "source_artifact_sha256": config["artifact_sha256"],
        "acceptance": "finite geometry checks only; sparse gaps are diagnostics, not a safety guarantee",
        **sphere_bound_metrics(mesh, spheres, depths),
    }
    return MeshRegionSphereResult(
        spheres, tuple(artifact.get("sphere_regions", ["frozen"] * len(spheres))), diagnostics
    )
