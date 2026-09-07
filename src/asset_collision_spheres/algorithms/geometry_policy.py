"""Opt-in routing layer; absence of geometry_policy NEVER selects this API."""

from __future__ import annotations

import time

from .convex_surface import (
    generate_convex_surface_spheres,
)
from .convex_surface import (
    normalize as normalize_convex,
)

VERSION = "geometry-policy-v4-experimental"


def evaluate_geometry_spheres(mesh, spheres, raw, *, seed=20260908, sample_count=12000):
    """Finite held-out target metrics; never run the old material evaluator."""
    import numpy as np

    from .hull_diagnostics import evaluate
    from .hull_structure import analyze, convex_hull, project
    from .original_surface import original_structure, surface_mesh

    config = normalize(raw)
    selected = config["geometry_policy"]
    if selected == "auto":
        from .geometry_router import route_geometry

        selected = route_geometry(mesh)["geometry_policy_selected"]
    target = convex_hull(mesh) if selected == "convex_hull" else surface_mesh(mesh)
    structure = (
        analyze(
            target, "auto" if config["shape_hint"] == "box" else config["shape_hint"]
        )
        if selected == "convex_hull"
        else original_structure(target, config["shape_hint"])
    )
    spheres = np.asarray(spheres, float)
    if (
        spheres.ndim != 2
        or spheres.shape[1] != 4
        or not 1 <= len(spheres) <= config["effective_max_spheres"]
        or not np.isfinite(spheres).all()
        or np.any(spheres[:, 3] <= 0)
        or np.any(spheres[:, 3] > 0.012 + 1e-12)
    ):
        raise ValueError("Invalid bounded [N,4] sphere set")
    if project(target, spheres[:, :3])[1].max() > max(
        1e-9, np.sqrt(target.area) * 1e-7
    ):
        raise ValueError("Spheres must be centered on the selected target surface")
    return {
        "target_geometry": selected,
        **evaluate(target, spheres, structure, seed=seed, sample_count=sample_count),
    }


def normalize(raw=None):
    raw = dict(raw or {})
    if raw.get("mode", "auto_geometry") != "auto_geometry":
        raise ValueError(
            "geometry_policy requires mode=auto_geometry; legacy modes are unchanged"
        )
    policy = raw.pop("geometry_policy", "auto")
    if policy not in {"auto", "convex_hull", "original_surface"}:
        raise ValueError(
            "geometry_policy must be auto, convex_hull or original_surface"
        )
    hint = raw.get("shape_hint", "auto")
    if hint not in {"auto", "open_box", "box", "generic"}:
        raise ValueError("shape_hint must be auto, open_box, box or generic")
    raw.update(mode="convex_surface", shape_hint="auto" if hint == "box" else hint)
    config = normalize_convex(raw)
    config.update(mode="auto_geometry", geometry_policy=policy, shape_hint=hint)
    return config


def generate_geometry_spheres(mesh, raw=None):
    started = time.perf_counter()
    config = normalize(raw)
    selected = config["geometry_policy"]
    routing = {
        "geometry_policy_requested": selected,
        "geometry_policy_selected": selected,
        "container_envelope_confidence": None,
        "evidence": {},
        "review_required": False,
        "reason": "explicit user override; not an automatic classification",
    }
    if selected == "auto":
        from .geometry_router import route_geometry

        routing = route_geometry(mesh)
        selected = routing["geometry_policy_selected"]
    routing_time = time.perf_counter() - started
    if selected == "convex_hull":
        delegated = {
            k: v
            for k, v in config.items()
            if k not in {"geometry_policy", "effective_max_spheres"}
        }
        delegated.update(
            mode="convex_surface",
            shape_hint="auto"
            if config["shape_hint"] == "box"
            else config["shape_hint"],
        )
        result = generate_convex_surface_spheres(mesh, delegated)
    else:
        from .original_surface import generate_original_surface_spheres

        result = generate_original_surface_spheres(mesh, config)
    result.diagnostics.update(
        geometry_routing=routing,
        geometry_policy_selected=selected,
        routing_version=VERSION,
        requested_config=config,
    )
    result.diagnostics["phase_times_s"] = {
        "routing": routing_time,
        "sampling_and_diagnostics": result.diagnostics["generation_time_s"],
    }
    result.diagnostics["generation_time_s"] = time.perf_counter() - started
    result.diagnostics["review_required"] |= routing["review_required"]
    return result
