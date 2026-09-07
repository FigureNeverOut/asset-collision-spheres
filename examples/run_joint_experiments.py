"""Isolated phase-ordered v2 evidence; never overwrites historical source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import trimesh

from asset_collision_spheres.api import generate_auto_geometry_spheres
from asset_collision_spheres.algorithms.box_features import (
    detect_box_features,
    evaluate_features,
)
from asset_collision_spheres.loaders.presets import read_preset
from asset_collision_spheres.loaders.usd import load_material_mesh
from asset_collision_spheres.paths import config_path, output_root, workspace_root
from asset_collision_spheres.preview.mesh import write_stage

OUT = output_root() / "structure_v2"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        + "\n"
    )


def freeze():
    root = workspace_root()
    sources = []
    for name in ("mug", "bowl", "box"):
        sources.extend(config_path(f"{name}", local=False).glob("*.json"))
    for relative in (
        "task2sim/runs/1e/mug_sphere_preview",
        "task2sim/runs/1e/mug_auto_geometry",
        "task2sim/runs/1e/mug_auto_geometry_32",
        "task2sim/runs/collision_ball_test/bowl_001_auto_geometry",
        "task2sim/runs/collision_ball_test/box_167_auto_geometry",
    ):
        for path in (root / relative).glob("*"):
            if path.suffix in {".json", ".usda", ".png"}:
                sources.append(path)
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
        if p.is_file()
    }
    path = OUT / "frozen_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(hashes.get(k) != v for k, v in previous.items()):
            raise RuntimeError(
                "Frozen source changed; do not silently refresh manifest"
            )
    else:
        dump(path, hashes)
    print(f"Frozen artifact/config hashes verified: {len(hashes)}", flush=True)


def synthetic(name):
    profile = np.array(
        [[0, 0], [0.12, 0], [0.12, 0.2], [0.113, 0.2], [0.113, 0.007], [0, 0.007]]
    )
    if name == "curved":
        return trimesh.creation.revolve(profile, sections=48)
    mesh = trimesh.creation.revolve(profile, sections=4)
    if name == "tray":
        # Align the diamond footprint before anisotropic scaling; otherwise this
        # fixture becomes a rhombus, not a rectangular shallow tray.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1])
        )
        mesh.apply_scale([1.8, 1.0, 0.25])
    elif name == "rotated":
        transform = trimesh.transformations.euler_matrix(0.41, -0.72, 0.18)
        transform[:3, 3] = [0.3, -0.7, 0.2]
        mesh.apply_transform(transform)
    elif name == "subdivided":
        mesh = mesh.subdivide()
    elif name == "tilted":
        transform = np.eye(4)
        transform[0, 2], transform[1, 2] = 0.12, 0.08
        mesh.apply_transform(transform)
    elif name == "rounded":
        # A smooth superelliptic cross-section is a boundary case, not a fake
        # sharp-corner box. Preserve material topology through a radial mapping.
        mesh = trimesh.creation.revolve(profile, sections=64)
        xy = mesh.vertices[:, :2]
        angle = np.arctan2(xy[:, 1], xy[:, 0])
        radius = np.linalg.norm(xy, axis=1)
        xy[:, 0] = radius * np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** 0.5
        xy[:, 1] = radius * np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** 0.5
        mesh.vertices[:, :2] = xy
    return mesh


def load_case(name):
    config = {
        "mode": "auto_geometry",
        "sphere_budget": 32,
        "max_outward_offset_m": 0.006,
        "seed": 5,
        "candidate_sample_count": 6000,
        "patch_count": 64,
        "balance_strength": 1.5,
        "probe_radius_m": 0.006,
        "probe_penetration_m": 0.0015,
        "repair_rounds": 1,
    }
    if name in {"box", "bowl"}:
        preset, config = read_preset(
            config_path(
                "box/box_167_auto_geometry.json"
                if name == "box"
                else "bowl/bowl_001_auto_geometry.json"
            )
        )
        mesh, _ = load_material_mesh(preset)
    elif name == "mug":
        from pxr import Usd
        from asset_collision_spheres.adapters.workspace.mug import mesh_data

        report = json.loads(
            (
                workspace_root()
                / "task2sim/runs/1e/mug_auto_geometry_32/auto_geometry.json"
            ).read_text()
        )
        mesh = mesh_data(Usd.Stage.Open(report["assets"]["mug"]), True)
    else:
        mesh = synthetic(name)
    return mesh, config


def features(names):
    for name in names:
        mesh, _ = load_case(name)
        result = detect_box_features(mesh, "open_box")
        out = OUT / "features" / name
        dump(out / "features.json", result.report())
        write_stage(out / "features.usda", mesh, [], result.report())
        print(
            name,
            "planes",
            len(result.planes),
            "edges",
            len(result.segments),
            "anchors",
            len(result.anchors),
            "confidence",
            result.confidence,
            flush=True,
        )


VARIANTS = {
    "A_legacy": {},
    "A0_single_control": {"shape_hint": "generic", "radius_mode": "single"},
    "B_candidates": {
        "shape_hint": "open_box",
        "radius_mode": "single",
        "feature_mode": "candidates",
    },
    "C_selection": {
        "shape_hint": "open_box",
        "radius_mode": "single",
        "feature_mode": "selection",
    },
    "D_features": {
        "shape_hint": "open_box",
        "radius_mode": "single",
        "feature_mode": "full",
    },
    "E_multi": {"shape_hint": "generic", "radius_mode": "multi"},
    "F_joint": {"shape_hint": "open_box", "radius_mode": "multi"},
    "G_auto": {"shape_hint": "auto", "radius_mode": "multi"},
    "H_adaptive": {
        "shape_hint": "auto",
        "radius_mode": "multi",
        "sphere_count_mode": "adaptive",
        "max_spheres": 64,
    },
    "I_extra_faces": {
        "shape_hint": "auto",
        "radius_mode": "multi",
        "sphere_count_mode": "adaptive",
        "max_spheres": 64,
        "extra_face_spheres": 8,
    },
    "J_scale_aware": {
        "shape_hint": "auto",
        "sphere_count_mode": "adaptive",
        "max_spheres": 64,
        "outward_offset_mode": "scale_aware",
        "max_outward_offset_m": None,
    },
    "K_adaptive_system": {
        "shape_hint": "auto",
        "sphere_count_mode": "adaptive",
        "max_spheres": None,
    },
    "L_scale_system": {
        "shape_hint": "auto",
        "sphere_count_mode": "adaptive",
        "max_spheres": None,
        "outward_offset_mode": "scale_aware",
        "max_outward_offset_m": None,
    },
    "M_scale_faces": {
        "shape_hint": "auto",
        "sphere_count_mode": "adaptive",
        "max_spheres": None,
        "outward_offset_mode": "scale_aware",
        "max_outward_offset_m": None,
        "extra_face_spheres": 8,
    },
}


def cost_calibration(names):
    """Shared-policy comparison, never a different policy chosen for each asset."""
    rows = []
    for name in names:
        subject, base = load_case(name)
        for budget in (32, 64):
            for weight in (0.0005, 0.0015, 0.003, 0.006):
                for power in (1, 2):
                    config = {
                        **base,
                        "sphere_budget": budget,
                        "shape_hint": "generic",
                        "system_policy": {
                            "outward_cost_weight": weight,
                            "outward_cost_power": power,
                        },
                    }
                    result = generate_auto_geometry_spheres(subject, config)
                    info = result.diagnostics
                    row = dict(
                        case=name,
                        budget=budget,
                        weight=weight,
                        power=power,
                        contact=info["diagnostics"]["contact_detection_fraction"],
                        radius_median_m=info["radius_median"],
                        large_fraction=info["selected_large_radius_fraction"],
                        free_intrusions=info["diagnostics"][
                            "free_probe_intrusions_beyond_allowance"
                        ],
                        seconds=info["generation_time_s"],
                    )
                    rows.append(row)
                    print(row, flush=True)
                    dump(OUT / "cost_calibration.json", rows)


def run_cases(names, variants, budgets):
    summary = {}
    for name in names:
        mesh, base = load_case(name)
        feature = detect_box_features(mesh, "open_box")
        for budget in budgets:
            for variant in variants:
                out = OUT / "ablations" / name / f"{variant}_{budget}"
                config = {**base, "sphere_budget": budget, **VARIANTS[variant]}
                print(f"RUN {name} {variant} {budget}", flush=True)
                result = generate_auto_geometry_spheres(mesh, config)
                info = dict(result.diagnostics)
                independent_feature = evaluate_features(mesh, result.spheres, feature)
                info["common_feature_evaluation"] = independent_feature
                dump(out / "generation.json", config)
                dump(
                    out / "result.json",
                    {
                        "config": config,
                        "spheres_object_local_m": result.spheres.tolist(),
                        "fit": info,
                    },
                )
                write_stage(
                    out / "preview.usda",
                    mesh,
                    result.spheres,
                    info.get("feature_diagnostics", independent_feature),
                )
                summary[f"{name}/{variant}/{budget}"] = {
                    "count": len(result.spheres),
                    "radius_quantiles_m": np.quantile(
                        result.spheres[:, 3], [0, 0.25, 0.5, 0.75, 1]
                    ).tolist(),
                    "outward_cap_m": info["max_outward_offset_m"],
                    "stop": info.get("count_stop_reason", "legacy_fixed"),
                    "contact": info["diagnostics"]["contact_detection_fraction"],
                    "anchors": independent_feature["corner_anchor_count"],
                    "supported": independent_feature["corner_anchor_supported_count"],
                    "edge_distribution": independent_feature["edge_distribution_score"],
                    "edge_max_gap_m": independent_feature["edge_max_gap_m"],
                    "free_intrusions": info["diagnostics"][
                        "free_probe_intrusions_beyond_allowance"
                    ],
                    "seconds": info["generation_time_s"],
                    "large_radius_fraction": info.get(
                        "selected_large_radius_fraction", 1
                    ),
                    "review_required": info["review_required"],
                    "missing_major_edges": independent_feature[
                        "major_edges_without_representation"
                    ],
                    "extra_added": info.get("extra_face_added", 0),
                    "extra_unfulfilled": info.get("extra_face_unfulfilled", 0),
                }
                print(summary[f"{name}/{variant}/{budget}"], flush=True)
                dump(
                    OUT / "ablation_summary.json",
                    {
                        **(
                            json.loads((OUT / "ablation_summary.json").read_text())
                            if (OUT / "ablation_summary.json").exists()
                            else {}
                        ),
                        **summary,
                    },
                )
    freeze()


def robustness():
    from asset_collision_spheres.geometry.validation import check_spheres

    rows = []

    def check(name, subject, config):
        result = generate_auto_geometry_spheres(subject, config)
        info = result.diagnostics
        check_spheres(
            subject,
            result.spheres,
            info["effective_hard_limit"],
            info["max_outward_offset_m"],
        )
        assert info["diagnostics"]["free_probe_intrusions_beyond_allowance"] == 0
        row = dict(
            case=name,
            count=len(result.spheres),
            seed=config["seed"],
            radius_median_m=info["radius_median"],
            contact=info["diagnostics"]["contact_detection_fraction"],
            feature_enabled=info["feature_diagnostics"]["enabled"],
            anchors=info["feature_diagnostics"]["corner_anchor_count"],
            supported=info["feature_diagnostics"]["corner_anchor_supported_count"],
            review_required=info["review_required"],
            seconds=info["generation_time_s"],
            stop=info["count_stop_reason"],
            cap_m=info["max_outward_offset_m"],
        )
        rows.append(row)
        print(row, flush=True)
        dump(OUT / "robustness.json", rows)

    for name in (
        "ideal",
        "rotated",
        "subdivided",
        "tray",
        "rounded",
        "tilted",
        "curved",
    ):
        subject, base = load_case(name)
        check(name, subject, {**base, "shape_hint": "auto", "sphere_budget": 32})
    for name in ("mug", "bowl", "box"):
        subject, base = load_case(name)
        for seed in (0, 5, 42):
            check(
                f"{name}/seed{seed}",
                subject,
                {**base, "shape_hint": "auto", "sphere_budget": 32, "seed": seed},
            )
        for scale in (0.5, 2.0):
            scaled = subject.copy()
            scaled.apply_scale(scale)
            equivalent = {**base, "shape_hint": "auto", "sphere_budget": 32}
            for key in (
                "max_outward_offset_m",
                "probe_radius_m",
                "probe_penetration_m",
            ):
                equivalent[key] *= scale
            check(f"{name}/scale{scale}/A_tolerances_scaled", scaled, equivalent)
            physical = {
                **base,
                "shape_hint": "auto",
                "sphere_count_mode": "adaptive",
                "max_spheres": 128,
            }
            check(f"{name}/scale{scale}/B_fixed_physical_tolerance", scaled, physical)
    freeze()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "freeze",
            "features",
            "ablations",
            "cost-calibration",
            "robustness",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--cases", nargs="+", default=["ideal", "box", "rotated"])
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[32, 64])
    args = parser.parse_args()
    freeze()
    if args.stage == "features":
        features(args.cases)
    elif args.stage == "ablations":
        run_cases(args.cases, args.variants, args.budgets)
    elif args.stage == "cost-calibration":
        cost_calibration(args.cases)
    elif args.stage == "robustness":
        robustness()
    elif args.stage == "all":
        features(
            [
                "ideal",
                "box",
                "rotated",
                "tray",
                "rounded",
                "tilted",
                "subdivided",
                "mug",
                "bowl",
                "curved",
            ]
        )
        run_cases(
            ["ideal", "box", "rotated"],
            [
                "A_legacy",
                "A0_single_control",
                "B_candidates",
                "C_selection",
                "D_features",
            ],
            [32, 64],
        )
        run_cases(
            ["mug", "bowl", "box"],
            ["A_legacy", "E_multi", "F_joint", "G_auto"],
            [32, 64],
        )
        run_cases(
            ["mug", "bowl", "box"],
            [
                "H_adaptive",
                "I_extra_faces",
                "J_scale_aware",
                "K_adaptive_system",
                "L_scale_system",
                "M_scale_faces",
            ],
            [64],
        )
        robustness()


if __name__ == "__main__":
    main()
