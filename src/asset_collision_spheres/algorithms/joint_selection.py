"""Opt-in v2: shared multi-radius/cost/count selector with optional mesh features.

The old public implementation is intentionally retained for unextended requests.
No category name participates in the material radius/cap/count policy.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, hstack

from ..geometry.material import _closest_point_exact, material_depth, sphere_clearance
from ..geometry.types import MeshRegionSphereResult
from ..geometry.validation import check_spheres, mesh_fingerprint, sphere_bound_metrics
from .box_features import (
    area_length,
    corner_witnesses,
    detect_box_features,
    evaluate_features,
    feature_samples,
)

VERSION = "auto-geometry-joint-v2-experimental"
NEW_KEYS = {
    "algorithm_variant",
    "shape_hint",
    "radius_mode",
    "feature_mode",
    "sphere_count_mode",
    "max_spheres",
    "backend_sphere_capacity",
    "outward_offset_mode",
    "extra_face_spheres",
    "system_policy",
}
POLICY = {
    "scale_alpha": 0.06,
    "absolute_outward_cap_m": 0.020,
    "system_max_spheres": 256,
    "initial_sphere_target": 8,
    "outward_cost_weight": 0.0015,
    "outward_cost_power": 2.0,
    "sphere_count_cost": 0.0015,
    "min_marginal_gain": 0.00015,
    "feature_gain_weight": 0.06,
    "feature_budget_fraction": 0.35,
    "max_iterations": 512,
    "generation_time_limit_s": 30.0,
}
RADIUS_LEVELS = (0.0, 0.35, 0.65, 1.0)


def is_extended(raw):
    return bool(NEW_KEYS.intersection(raw))


def normalize(mesh, raw):
    from . import auto_geometry as old

    raw = dict(raw)
    if raw.get("algorithm_variant", "joint_v2") != "joint_v2":
        raise ValueError("algorithm_variant must be joint_v2")
    if not isinstance(raw.get("system_policy", {}), dict):
        raise ValueError("system_policy must be an object")
    policy = {**POLICY, **raw.get("system_policy", {})}
    if set(policy) - set(POLICY):
        raise ValueError("Unknown system_policy setting")
    for name, value in policy.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid system_policy {name}")
    for name in ("system_max_spheres", "initial_sphere_target", "max_iterations"):
        if type(policy[name]) is not int or policy[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if policy["system_max_spheres"] > 256 or policy["max_iterations"] > 4096:
        raise ValueError("System resource limit exceeds supported safety bound")
    if policy["outward_cost_power"] < 1:
        raise ValueError("outward_cost_power must be at least 1")
    if (
        not 0 < policy["feature_budget_fraction"] <= 0.5
        or policy["generation_time_limit_s"] <= 0
    ):
        raise ValueError(
            "Feature protection must reserve at least half the budget for ordinary geometry"
        )
    mode = raw.get(
        "sphere_count_mode", "fixed" if "sphere_budget" in raw else "adaptive"
    )
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("sphere_count_mode must be fixed or adaptive")
    limits = [policy["system_max_spheres"]]
    for key in ("max_spheres", "backend_sphere_capacity"):
        if raw.get(key) is not None:
            if type(raw[key]) is not int or raw[key] < 1:
                raise ValueError(f"{key} must be a positive integer or null")
            limits.append(raw[key])
    hard_limit = min(limits)
    if mode == "fixed":
        target = raw.get("sphere_budget")
        if type(target) is not int or not 1 <= target <= hard_limit:
            raise ValueError(
                "fixed sphere_budget must fit max_spheres/system/backend capacity"
            )
        hard_limit = target  # supplements also cannot silently exceed a fixed budget
    else:
        target = min(policy["initial_sphere_target"], hard_limit)
    extra = raw.get("extra_face_spheres", 0)
    if type(extra) is not int or extra < 0 or extra > 256:
        raise ValueError("extra_face_spheres must be an integer in [0,256]")
    # An explicit old cap always remains a cap, including with scale_aware set.
    outward_mode = raw.get("outward_offset_mode", "fixed")
    if outward_mode not in {"fixed", "scale_aware"}:
        raise ValueError("outward_offset_mode must be fixed or scale_aware")
    length = area_length(mesh)
    derived = min(policy["scale_alpha"] * length, policy["absolute_outward_cap_m"])
    explicit = raw.get("max_outward_offset_m")
    if explicit is None and outward_mode == "fixed":
        raise ValueError("fixed outward mode requires max_outward_offset_m")
    if explicit is not None and (
        isinstance(explicit, bool)
        or not isinstance(explicit, (int, float))
        or not np.isfinite(explicit)
        or explicit < 0
    ):
        raise ValueError("max_outward_offset_m must be finite and nonnegative")
    cap = (
        explicit
        if outward_mode == "fixed"
        else min(derived, explicit)
        if explicit is not None
        else derived
    )
    config = old.normalize_config(
        mesh,
        {
            **{k: v for k, v in raw.items() if k not in NEW_KEYS},
            "sphere_budget": hard_limit if mode == "adaptive" else target,
            "max_outward_offset_m": cap,
        },
    )
    for key, choices, default in (
        ("shape_hint", {"auto", "open_box", "generic"}, "auto"),
        ("radius_mode", {"single", "multi"}, "multi"),
        ("feature_mode", {"off", "candidates", "selection", "full"}, "full"),
    ):
        config[key] = raw.get(key, default)
        if config[key] not in choices:
            raise ValueError(f"Invalid {key}")
    config.update(
        algorithm_variant="joint_v2",
        sphere_count_mode=mode,
        max_spheres=raw.get("max_spheres"),
        backend_sphere_capacity=raw.get("backend_sphere_capacity"),
        effective_hard_limit=hard_limit,
        initial_sphere_target=target,
        extra_face_spheres=extra,
        outward_offset_mode=outward_mode,
        system_policy=policy,
        scale_reference_m=length,
        scale_definition="sqrt(material_surface_area / (2*pi)); area includes inner/outer walls; rigid and triangulation invariant",
        cap_from_scale_m=derived,
    )
    return config


def legacy_config(config):
    # A single allow-list avoids passing v2 diagnostics back to the legacy schema.
    keys = {
        "mode",
        "sphere_budget",
        "max_outward_offset_m",
        "seed",
        "candidate_sample_count",
        "patch_count",
        "repair_rounds",
        "probe_radius_m",
        "probe_penetration_m",
        "balance_strength",
        "free_space_weight",
        "redundancy_weight",
    }
    return {k: v for k, v in config.items() if k in keys}


def additional_centres(mesh, features):
    points, _ = feature_samples(mesh, features)
    neighbourhoods = corner_witnesses(mesh, features)
    points = np.r_[features.anchors, neighbourhoods.reshape(-1, 3), points]
    if not len(points):
        return np.empty((0, 3)), np.empty(0), np.empty(0, int)
    _, _, faces = _closest_point_exact(mesh, points)
    # Surface normal plus incident plane directions; all proposals still require
    # original-mesh membership. No averaged-normal assumption is trusted.
    directions = [mesh.face_normals[faces]]
    pn = np.asarray([p["normal"] for p in features.planes])
    if len(pn):
        pc = np.asarray([p["center_m"] for p in features.planes])
        proximity = np.abs(np.einsum("npi,pi->np", points[:, None] - pc, pn))
        closest = np.argsort(proximity, axis=1)[:, :2]
        directions.extend([pn[closest[:, 0]], pn[closest[:, 1]]])
        combined = pn[closest[:, 0]] + pn[closest[:, 1]]
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
        directions.append(combined)
    proposals = np.concatenate(
        [
            (
                points[:, None]
                - direction[:, None]
                * (features.scale * np.array([0.001, 0.003, 0.007, 0.015]))[
                    None, :, None
                ]
            ).reshape(-1, 3)
            for direction in directions
        ]
    )
    depths = material_depth(mesh, proposals, exact_distance=True)
    keep = np.flatnonzero(depths > features.scale * 1e-6)
    # Retain shallow and deep alternatives; cap work uniformly, not by asset.
    keep = keep[:: max(1, int(np.ceil(len(keep) / 2400)))]
    proposals, depths = proposals[keep], depths[keep]
    if not len(proposals):
        return proposals, depths, np.empty(0, int)
    _, _, faces = _closest_point_exact(mesh, proposals)
    return proposals, depths, faces


def generate(mesh, raw):
    from . import auto_geometry as old

    started = time.perf_counter()
    config = normalize(mesh, raw)
    policy = config["system_policy"]
    cap = config["max_outward_offset_m"]
    hard_limit = config["effective_hard_limit"]
    features = detect_box_features(mesh, config["shape_hint"])
    feature_time = time.perf_counter() - started
    base = legacy_config(config)
    # Candidate deficiency is reported by this selector; legacy centre generation
    # still checks all material assumptions but need not produce the full capacity.
    pool_cfg = {**base, "sphere_budget": 1}
    centres, max_radii, surface, faces, origins, witnesses = old.candidate_pool(
        mesh, pool_cfg
    )
    depth = material_depth(mesh, centres, exact_distance=True)
    origin_faces = faces[origins]
    source_types = np.full(len(centres), "surface", dtype="U16")
    if features.enabled and config["feature_mode"] in {"candidates", "full"}:
        extra_c, extra_d, extra_f = additional_centres(mesh, features)
        centres, depth, origin_faces = (
            np.r_[centres, extra_c],
            np.r_[depth, extra_d],
            np.r_[origin_faces, extra_f],
        )
        source_types = np.r_[source_types, np.full(len(extra_c), "feature")]
    # Mutually exclusive groups also collapse duplicate centres across sources.
    tree = cKDTree(centres)
    used = np.zeros(len(centres), bool)
    unique = []
    for i in range(len(centres)):
        if not used[i]:
            unique.append(i)
            used[tree.query_ball_point(centres[i], features.scale * 0.00025)] = True
    centres, depth, origin_faces, source_types = (
        centres[unique],
        depth[unique],
        origin_faces[unique],
        source_types[unique],
    )
    levels = np.asarray([1.0] if config["radius_mode"] == "single" else RADIUS_LEVELS)
    if cap == 0:
        levels = np.asarray([0.0])
    group = np.repeat(np.arange(len(centres)), len(levels))
    expansion = np.tile(levels * cap, len(centres))
    radii = depth[group] + expansion - old.characteristic_length(mesh) * 5e-6
    valid = radii > 0
    group, radii, expansion = group[valid], radii[valid], expansion[valid]
    pool = np.c_[centres[group], radii]
    labels, area, components, _ = old.local_patches(mesh, faces, config["patch_count"])
    tags = labels[faces]
    cover, neighbours = old.incidence(
        pool[:, :3], radii + config["probe_radius_m"], witnesses
    )
    neighbours = [np.asarray(n, int) for n in neighbours]
    candidate_time = time.perf_counter() - started - feature_time
    free_points, confirmed, beyond, probe_radius = old.free_samples(
        mesh, surface, faces, base, seed=config["seed"] + 31
    )
    if beyond.any():
        free_cover, _ = old.incidence(
            pool[:, :3], radii + probe_radius, free_points[beyond]
        )
        free_cost = np.asarray(free_cover.sum(axis=1)).ravel() / beyond.sum()
    else:
        free_cost = np.zeros(len(pool))
    outward_cost = (
        policy["outward_cost_weight"]
        * (expansion / max(cap, 1e-12)) ** policy["outward_cost_power"]
    )
    if config["radius_mode"] == "single":
        outward_cost[:] = 0  # controlled single-radius feature ablation
    edge_points, edge_ids = feature_samples(mesh, features)
    anchors = features.anchors if features.enabled else np.empty((0, 3))
    fp = np.r_[anchors, edge_points]
    fm, _ = old.incidence(pool[:, :3], radii + features.scale * 0.006, fp)
    if len(anchors):
        targets = corner_witnesses(mesh, features)
        # One candidate represents a small corner neighbourhood, not merely an
        # isolated point. The union evaluator also permits cooperating balls.
        supports = np.empty((len(pool), len(anchors)), bool)
        for i, points in enumerate(targets):
            supports[:, i] = (
                np.linalg.norm(pool[:, None, :3] - points, axis=2)
                <= radii[:, None] + 0.001 * features.scale
            ).all(axis=1)
        fm = hstack([csr_matrix(supports), fm[:, len(anchors) :]], format="csr")
    f_neighbours = [
        fm.indices[fm.indptr[i] : fm.indptr[i + 1]] for i in range(len(pool))
    ]
    f_neighbours = [np.asarray(n, int) for n in f_neighbours]
    feature_active = features.enabled and config["feature_mode"] in {
        "selection",
        "full",
    }
    fw = np.r_[
        np.full(len(anchors), 1 / max(len(anchors), 1)),
        np.full(len(edge_points), 1 / max(len(edge_points), 1)),
    ]
    fw *= policy["feature_gain_weight"] if feature_active else 0
    # A long edge's middle third has its own diminishing-return target. Merely
    # covering both endpoints cannot collect all the distribution reward.
    edge_bins = np.empty(len(edge_points), int)
    for eid in np.unique(edge_ids):
        positions = np.flatnonzero(edge_ids == eid)
        for third, part in enumerate(np.array_split(positions, 3)):
            edge_bins[part] = int(eid) * 3 + third
    bin_count = len(features.segments) * 3 if features.enabled else 0
    mapping = csr_matrix(
        (np.ones(len(edge_points)), (np.arange(len(edge_points)), edge_bins)),
        shape=(len(edge_points), bin_count),
    )
    bm = (fm[:, len(anchors) :] @ mapping).astype(bool).astype(float).tocsr()
    bin_weight = (
        policy["feature_gain_weight"] / max(bin_count, 1) if feature_active else 0
    )
    fw[len(anchors) :] *= 0.5
    bin_weight *= 0.5
    hits = np.zeros(len(witnesses), int)
    fhits = np.zeros(len(fp), int)
    selected = {}  # centre group -> one radius variant
    protected = np.empty(0, int)
    trace = []
    area_factor = np.clip(np.sqrt(area.mean() / np.maximum(area, 1e-20)), 0.5, 2)
    patch_samples = np.bincount(tags, minlength=len(area))
    selected_sources = {}

    def weights():
        covered = np.bincount(tags, weights=hits > 0, minlength=len(area))
        boost = (
            config["balance_strength"]
            * area_factor
            * (1 - covered / np.maximum(patch_samples, 1)) ** 2
        )
        if config["mode"] == "global":
            boost[:] = 0
        return (1 + boost[tags]) / len(hits)

    def change(index, reason):
        nonlocal protected
        old_index = selected.get(int(group[index]))
        if old_index is not None:
            hits[neighbours[old_index]] -= 1
            fhits[f_neighbours[old_index]] -= 1
        selected[int(group[index])] = index
        hits[neighbours[index]] += 1
        fhits[f_neighbours[index]] += 1
        selected_sources.setdefault(int(group[index]), reason)

    def scores(allow_new=True, face_only=False):
        w = weights()
        gain = np.asarray(cover @ ((hits == 0) * w)).ravel()
        redundancy = (
            config["redundancy_weight"]
            * np.asarray(cover @ (hits > 0).astype(float)).ravel()
            / len(hits)
        )
        fg = (
            np.asarray(fm @ ((fhits == 0) * fw)).ravel()
            if len(fp) and not face_only
            else np.zeros(len(pool))
        )
        bin_hits = np.bincount(
            edge_bins, weights=fhits[len(anchors) :] > 0, minlength=bin_count
        )
        if feature_active and not face_only:
            fg += np.asarray(bm @ (bin_hits == 0).astype(float)).ravel() * bin_weight
        score = (
            gain
            + fg
            - redundancy
            - outward_cost
            - policy["sphere_count_cost"]
            - config["free_space_weight"] * free_cost
        )
        if not allow_new:
            score[:] = -np.inf
        if face_only:
            score[source_types[group] != "surface"] = -np.inf
            # Supplement main faces/patch interiors, away from feature witnesses.
            if len(fp):
                near = cKDTree(fp).query(pool[:, :3])[0] < features.scale * 0.04
                score[near] = -np.inf
        for g, old_index in selected.items():
            versions = np.flatnonzero(group == g)
            score[versions] = -np.inf
            if face_only:
                continue
            old_n, old_f = neighbours[old_index], f_neighbours[old_index]
            for v in versions:
                if radii[v] <= radii[old_index] + 1e-12:
                    continue
                # Enlargements preserve containment at a common centre. Only
                # newly covered witnesses count; old self-overlap is not a cost.
                added = np.setdiff1d(neighbours[v], old_n, assume_unique=False)
                extra_f = np.setdiff1d(f_neighbours[v], old_f, assume_unique=False)
                score[v] = (w[added] * (hits[added] == 0)).sum() + (
                    fw[extra_f] * (fhits[extra_f] == 0)
                ).sum()
                score[v] -= outward_cost[v] - outward_cost[old_index]
                score[v] -= (
                    config["redundancy_weight"] * (hits[added] > 0).sum() / len(hits)
                )
                score[v] -= config["free_space_weight"] * (
                    free_cost[v] - free_cost[old_index]
                )
                score[v] += (bm[v] @ (bin_hits == 0).astype(float)).item() * bin_weight
        return score

    selection_started = time.perf_counter()
    fixed = config["sphere_count_mode"] == "fixed"
    target = config["initial_sphere_target"]
    primary_limit = target if fixed else hard_limit
    # At most 35% of the available initial structural budget is reserved. A
    # protected anchor means selected support may be replaced, not silently lost.
    protect_limit = min(
        primary_limit,
        int(max(target, min(32, primary_limit)) * policy["feature_budget_fraction"]),
    )
    if feature_active and len(anchors):
        for _ in range(protect_limit):
            missing = np.r_[
                fhits[: len(anchors)] == 0, np.zeros(len(edge_points), bool)
            ]
            support_gain = np.asarray(fm @ missing.astype(float)).ravel()
            for g in selected:
                support_gain[group == g] = -1
            if support_gain.max(initial=0) <= 0:
                break
            options = np.flatnonzero(support_gain == support_gain.max())
            index = int(options[np.argmax(scores()[options])])
            change(index, "corner_protection")
        protected = np.flatnonzero(fhits[: len(anchors)] > 0)
    protected_bins = np.empty(0, int)
    if feature_active and bin_count:
        for _ in range(max(0, protect_limit - len(selected))):
            bin_hits = np.bincount(
                edge_bins, weights=fhits[len(anchors) :] > 0, minlength=bin_count
            )
            missing_middle = (bin_hits == 0) & (np.arange(bin_count) % 3 == 1)
            middle_gain = np.asarray(bm @ missing_middle.astype(float)).ravel()
            for g in selected:
                middle_gain[group == g] = -1
            if middle_gain.max(initial=0) <= 0:
                break
            options = np.flatnonzero(middle_gain == middle_gain.max())
            change(
                int(options[np.argmax(scores()[options])]),
                "edge_distribution_protection",
            )
        bin_hits = np.bincount(
            edge_bins, weights=fhits[len(anchors) :] > 0, minlength=bin_count
        )
        protected_bins = np.flatnonzero(
            (bin_hits > 0) & (np.arange(bin_count) % 3 == 1)
        )
    feature_seed_count = len(selected)
    minimum_primary_count = min(
        primary_limit,
        max(
            target, int(np.ceil(feature_seed_count / policy["feature_budget_fraction"]))
        ),
    )
    stop = "iteration_limit"
    for iteration in range(policy["max_iterations"]):
        if (
            selected
            and time.perf_counter() - started > policy["generation_time_limit_s"]
        ):
            stop = "time_limit"
            break
        score = scores(allow_new=len(selected) < primary_limit)
        index = int(np.argmax(score))
        if not np.isfinite(score[index]):
            stop = (
                "capacity_limit"
                if len(selected) >= primary_limit
                else "candidate_deficiency"
            )
            break
        if (
            score[index] <= policy["min_marginal_gain"]
            and (not fixed or len(selected) >= target)
            and len(selected) >= minimum_primary_count
        ):
            stop = (
                "capacity_limit" if len(selected) >= primary_limit else "marginal_gain"
            )
            break
        action = "enlarge" if int(group[index]) in selected else "add"
        trace.append(
            {
                "iteration": iteration,
                "action": action,
                "candidate": index,
                "marginal_score": float(score[index]),
            }
        )
        change(index, "ordinary_selection")
    selection_time = time.perf_counter() - selection_started
    repair_started = time.perf_counter()
    repairs = []

    # Bounded one-out/two-in trials allow a large ball to be replaced by smaller
    # alternatives. Protected anchors survive every accepted swap.
    def utility(indices):
        counts = np.zeros(len(hits), int)
        fc = np.zeros(len(fp), int)
        for index in indices:
            counts[neighbours[index]] += 1
            fc[f_neighbours[index]] += 1
        fractions = np.bincount(
            tags, weights=counts > 0, minlength=len(area)
        ) / np.maximum(patch_samples, 1)
        return (
            float((counts > 0).mean())
            + (config["balance_strength"] if config["mode"] == "auto_geometry" else 0)
            * float(np.sum(area * np.sqrt(fractions)) / area.sum())
            + float(fw @ (fc > 0))
            - float(outward_cost[indices].sum())
            - policy["sphere_count_cost"] * len(indices)
            + bin_weight
            * float(
                np.sum(
                    np.bincount(
                        edge_bins, weights=fc[len(anchors) :] > 0, minlength=bin_count
                    )
                    > 0
                )
            )
            - config["redundancy_weight"]
            * float(np.maximum(counts - 1, 0).sum() / len(hits))
            - config["free_space_weight"] * float(free_cost[indices].sum())
        ), fc

    for _ in range(config["repair_rounds"]):
        if (
            time.perf_counter() - started > policy["generation_time_limit_s"]
            or not selected
        ):
            break
        original = dict(selected)
        original_sources = dict(selected_sources)
        old_value, _ = utility(list(selected.values()))
        remove = min(
            selected,
            key=lambda g: (
                old_value - utility([v for h, v in selected.items() if h != g])[0]
            ),
        )
        removed = selected.pop(remove)
        hits[neighbours[removed]] -= 1
        fhits[f_neighbours[removed]] -= 1
        limit = 1 if fixed else min(2, hard_limit - len(selected))
        for _step in range(limit):
            score = scores(allow_new=True)
            index = int(np.argmax(score))
            if not np.isfinite(score[index]):
                break
            if fixed or score[index] > policy["min_marginal_gain"]:
                change(index, "repair")
        new_value, new_fc = utility(list(selected.values()))
        new_bins = np.bincount(
            edge_bins, weights=new_fc[len(anchors) :] > 0, minlength=bin_count
        )
        accepted = (
            new_value > old_value + 1e-9
            and np.all(new_fc[protected] > 0)
            and np.all(new_bins[protected_bins] > 0)
            and (not fixed or len(selected) == len(original))
        )
        repairs.append(
            {"accepted": bool(accepted), "old_score": old_value, "new_score": new_value}
        )
        if not accepted:
            selected, selected_sources = original, original_sources
            hits[:] = 0
            fhits[:] = 0
            for index in selected.values():
                hits[neighbours[index]] += 1
                fhits[f_neighbours[index]] += 1
    repair_time = time.perf_counter() - repair_started
    supplement = 0
    # No refill/re-randomization: append only surface-origin interior candidates.
    for _ in range(config["extra_face_spheres"]):
        if (
            len(selected) >= hard_limit
            or time.perf_counter() - started > policy["generation_time_limit_s"]
        ):
            break
        score = scores(face_only=True)
        if not np.isfinite(score).any():
            break
        index = int(np.argmax(score))
        change(index, "extra_face")
        supplement += 1
    indices = np.asarray(list(selected.values()), int)
    if not len(indices):
        raise ValueError("candidate deficiency: no valid sphere selected")
    spheres = pool[indices]
    selected_depth = check_spheres(mesh, spheres, hard_limit, cap)
    diagnostic_started = time.perf_counter()
    ordinary = old.evaluate_auto_geometry_spheres(
        mesh, spheres, {**base, "sphere_budget": hard_limit}
    )
    feature_report = evaluate_features(mesh, spheres, features)
    detected = sphere_clearance(witnesses, spheres, config["probe_radius_m"]) <= 0
    review_points, review_faces = old.trimesh.sample.sample_surface(
        mesh, 2048, seed=20260909
    )
    review_witnesses = review_points + mesh.face_normals[review_faces] * (
        config["probe_radius_m"] - config["probe_penetration_m"]
    )
    review_hits = (
        sphere_clearance(review_witnesses, spheres, config["probe_radius_m"]) <= 0
    )
    missing_planes = [
        p["id"]
        for p in features.planes
        if features.enabled
        and np.any(np.isin(review_faces, p["face_ids"]))
        and not np.any(review_hits[np.isin(review_faces, p["face_ids"])])
    ]
    feature_report["planar_patches_without_contact_samples"] = missing_planes
    # Detection keeps small useful fragments (>=0.8% area); do not call every
    # scan fragment an entire major face. Report both, conservatively reviewing
    # omissions in either set without conflating them with semantic walls.
    feature_report["major_plane_minimum_area_fraction"] = 0.03
    feature_report["major_planes_without_contact_samples"] = [
        p["id"]
        for p in features.planes
        if p["id"] in missing_planes and p["area_m2"] >= 0.03 * mesh.area
    ]
    feature_report["major_plane_review_seed"] = 20260909
    feature_report["major_plane_review_sample_count"] = 2048
    feature_report["anchors_without_single_candidate_support"] = np.flatnonzero(
        np.asarray(fm[:, : len(anchors)].sum(axis=0)).ravel() == 0
    ).tolist()
    missing_components = sorted(
        set(components.tolist())
        - set(components[origin_faces[group[indices]]].tolist())
    )
    exp = np.maximum(spheres[:, 3] - selected_depth, 0)
    quantiles = np.quantile(spheres[:, 3], [0, 0.25, 0.5, 0.75, 1])
    patches = [
        {
            "id": int(i),
            "area_m2": float(a),
            "contact_coverage": float(detected[tags == i].mean())
            if np.any(tags == i)
            else 0.0,
            "selected_origin_count": int(
                np.sum(labels[origin_faces[group[indices]]] == i)
            ),
        }
        for i, a in enumerate(area)
    ]
    candidate_digest = hashlib.sha256(pool.astype("<f8").tobytes()).hexdigest()
    report = {
        "algorithm_version": VERSION,
        "mode": config["mode"],
        "config": config,
        "mesh_sha256": mesh_fingerprint(mesh),
        "candidate_pool_sha256": candidate_digest,
        "actual_count": len(spheres),
        "actual_sphere_count": len(spheres),
        "requested_fixed_count": target if fixed else None,
        "feature_seed_count": feature_seed_count,
        "minimum_primary_count_after_feature_reservation": minimum_primary_count,
        "backend_sphere_capacity": config["backend_sphere_capacity"],
        "effective_hard_limit": hard_limit,
        "count_stop_reason": stop,
        "candidates_considered": len(pool),
        "candidate_count": len(pool),
        "centre_count": len(centres),
        "radius_range_m": [float(quantiles[0]), float(quantiles[-1])],
        **{
            name: float(value)
            for name, value in zip(
                [
                    "radius_min",
                    "radius_p25",
                    "radius_median",
                    "radius_p75",
                    "radius_max",
                ],
                quantiles,
            )
        },
        "max_outward_offset_m": cap,
        "outward_expansion_min": float(exp.min()),
        "outward_expansion_median": float(np.median(exp)),
        "outward_expansion_max": float(exp.max()),
        "selected_large_radius_fraction": float(np.mean(exp >= cap * 0.95))
        if cap > 0
        else 0.0,
        "selected_center_group_ids": group[indices].tolist(),
        "selected_source_types": source_types[group[indices]].tolist(),
        "selected_outward_expansions_m": exp.tolist(),
        "selection_trace": trace,
        "repair_log": repairs,
        "radius_levels": levels.tolist(),
        "selected_candidates": [
            {
                "center_m": centres[group[index]].tolist(),
                "radius_m": float(radii[index]),
                "material_depth_m": float(depth[group[index]]),
                "outward_expansion_m": float(expansion[index]),
                "source_type": str(source_types[group[index]]),
                "selection_reason": selected_sources[int(group[index])],
                "patch_id": int(labels[origin_faces[group[index]]]),
                "corner_support_ids": f_neighbours[index][
                    f_neighbours[index] < len(anchors)
                ].tolist(),
                "covered_contact_witness_count": len(neighbours[index]),
                "free_space_cost": float(free_cost[index]),
                "outward_cost": float(outward_cost[index]),
            }
            for index in indices
        ],
        "protected_corner_ids": protected.tolist(),
        "extra_face_requested": config["extra_face_spheres"],
        "extra_face_added": supplement,
        "protected_edge_middle_bin_ids": protected_bins.tolist(),
        "extra_face_unfulfilled": config["extra_face_spheres"] - supplement,
        "feature_diagnostics": feature_report,
        "surface_contact_diagnostics": ordinary,
        "diagnostics": ordinary,
        "free_space_diagnostics": {
            k: v
            for k, v in ordinary.items()
            if "free_probe" in k or "allowed_band" in k
        },
        "local_plane_footprint_median_m": float(
            np.median(np.sqrt(np.maximum(spheres[:, 3] ** 2 - selected_depth**2, 0)))
        ),
        "footprint_note": "sqrt(r^2-d^2), tangent-plane approximation only; outward cost is a proxy, not exact outside volume",
        "patches": patches,
        "missing_material_components": missing_components,
        "review_required": bool(
            missing_components
            or missing_planes
            or feature_report["unsupported_corner_ids"]
            or feature_report["major_edges_without_representation"]
            or len(spheres) < minimum_primary_count
        ),
        "validation_status": "finite_checks_passed",
        "used_convex_hull": False,
        "fallback_used": not features.enabled and config["shape_hint"] != "generic",
        "feature_detection_time_s": feature_time,
        "candidate_time_s": candidate_time,
        "selection_time_s": selection_time,
        "repair_time_s": repair_time,
        "diagnostic_time_s": time.perf_counter() - diagnostic_started,
        "generation_time_s": diagnostic_started - started,
        "time_limit_scope": "checked between selection/repair/supplement iterations, not a hard interrupt of geometry queries or diagnostics",
        **{
            k: ordinary[k]
            for k in (
                "surface_coverage",
                "surface_gap_mean_m",
                "surface_gap_p95_m",
                "max_uncovered_gap_m",
                "ordinary_sample_count",
            )
        },
        **sphere_bound_metrics(mesh, spheres, selected_depth),
    }
    return MeshRegionSphereResult(
        spheres,
        tuple(f"patch_{labels[origin_faces[g]]}" for g in group[indices]),
        report,
    )
