"""Full-material mesh adapter. No automatic convex fallback or semantic guesses."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from ..geometry.types import MeshRegionSphereResult
from ..geometry.material import material_depth, sphere_clearance
from ..geometry.validation import (characteristic_length, sphere_bound_metrics, mesh_fingerprint, check_material_mesh, check_spheres)

ADAPTER_VERSION = "full-material-native-v2"












def prepare_request(raw, *, capacity, seed):
    """Resolve immutable artifact content before cache lookup (replacement invalidates)."""
    config = dict(raw)
    if "geometry_policy" in config:
        from ..algorithms.geometry_policy import normalize, VERSION
        if config.get("backend_sphere_capacity") not in (None, capacity):
            raise ValueError("backend_sphere_capacity comes from the actual backend")
        config.setdefault("mode", "auto_geometry")
        config["backend_sphere_capacity"] = capacity
        config.setdefault("seed", seed)
        resolved = normalize(config)
        digest = hashlib.sha256(json.dumps([VERSION, resolved], sort_keys=True).encode()).hexdigest()
        return config, digest
    if config.get("mode") == "convex_surface":
        from ..algorithms.convex_surface import normalize
        if config.get("backend_sphere_capacity") not in (None, capacity):
            raise ValueError("backend_sphere_capacity comes from the actual backend")
        config["backend_sphere_capacity"] = capacity
        config.setdefault("seed", seed)
        normalize(config)
        # Version only this new path; historical material cache keys unchanged.
        digest = hashlib.sha256(json.dumps(["convex-surface-v3-experimental", config], sort_keys=True).encode()).hexdigest()
        return config, digest
    config.setdefault("mode", "auto_geometry")
    config.setdefault("sphere_budget", capacity)
    config.setdefault("seed", seed)
    from ..algorithms.joint_selection import is_extended
    if config["mode"] != "precomputed" and is_extended(config):
        if config.get("backend_sphere_capacity") not in (None, capacity):
            raise ValueError("backend_sphere_capacity is supplied by the actual backend, not the asset")
        config["backend_sphere_capacity"] = capacity
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
    if "geometry_policy" in config:
        from ..algorithms.geometry_policy import generate_geometry_spheres
        result = generate_geometry_spheres(mesh, config)
        if result.diagnostics.get("connectivity_fairness", {}).get("missing_components"):
            raise ValueError("Original surface has unrepresented connected components; increase cap or review mesh before native attachment")
        return result
    if config["mode"] == "convex_surface":
        from ..algorithms.convex_surface import generate_convex_surface_spheres
        return generate_convex_surface_spheres(mesh, config)
    if config["mode"] != "precomputed":
        from ..algorithms.auto_geometry import generate_auto_geometry_spheres

        result = generate_auto_geometry_spheres(mesh, config)
        if result.diagnostics["missing_material_components"]:
            raise ValueError(
                "Entire material components have no sphere representation; inspect the diagnostic output "
                "and budget before native attachment. Ordinary contact misses are not this gate."
            )
        features = result.diagnostics.get("feature_diagnostics", {})
        if features.get("enabled") and (
            features.get("unsupported_corner_ids")
            or features.get("major_edges_without_representation")
            or features.get("major_planes_without_contact_samples")
        ):
            raise ValueError(
                "Important detected structure lacks representation; review the saved generator "
                "diagnostics and budget/cap before native attachment. Ordinary sparse gaps and "
                "uncertain detection with generic fallback do not trigger this gate."
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
