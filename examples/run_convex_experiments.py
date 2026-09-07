"""Frozen-history reference + same-hull/count/radius/evaluation sampling ablations."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from asset_collision_spheres.api import generate_convex_surface_spheres
from asset_collision_spheres.algorithms.convex_surface import stats
from asset_collision_spheres.algorithms.hull_structure import convex_hull, analyze
from asset_collision_spheres.algorithms.hull_diagnostics import (
    evaluate,
    obstacle_approaches,
)
from asset_collision_spheres.preview.convex import write_stage
from asset_collision_spheres.paths import output_root
from run_joint_experiments import load_case, synthetic
from freeze_convex_reference import freeze

OUT = output_root() / "convex_v3"
CASES = (
    "cube",
    "rectangular",
    "box",
    "rotated_box",
    "tray",
    "rounded",
    "bowl",
    "smooth_bowl",
    "mug",
    "subdivided",
)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def case_mesh(name):
    if name == "cube":
        return trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    if name in {"rectangular", "subdivided"}:
        mesh = trimesh.creation.box(extents=[0.4, 0.2, 0.12])
        return mesh.subdivide() if name == "subdivided" else mesh
    if name == "smooth_bowl":
        angles = np.linspace(0, np.pi / 2, 24)
        outer = np.column_stack((0.15 * np.sin(angles), 0.12 * (1 - np.cos(angles))))
        inner = np.column_stack(
            (0.143 * np.sin(angles[::-1]), 0.007 + 0.113 * (1 - np.cos(angles[::-1])))
        )
        return trimesh.creation.revolve(np.vstack((outer, inner)), sections=96)
    if name == "rotated_box":
        mesh, _ = load_case("box")
        transform = trimesh.transformations.euler_matrix(0.43, -0.68, 0.9)
        transform[:3, 3] = [0.4, -0.8, 0.3]
        mesh.apply_transform(transform)
        return mesh
    if name in {"tray", "rounded"}:
        return synthetic(name)
    return load_case(name)[0]


def row(name, budget, variant, result):
    d = result.diagnostics
    surface = d["surface_uniformity"]["site_distance_m"]
    loops = d["feature_diagnostics"]["rim_loops"]
    return {
        "case": name,
        "max_spheres": budget,
        "variant": variant,
        "actual_count": len(result.spheres),
        "strategy": d["sampling_strategy"],
        "radius_m": d["radius_m"],
        "site_distance_m": surface,
        "corner_support": d["feature_diagnostics"]["corners"],
        "rim_site_counts": [p["site_count_on_feature"] for p in loops],
        "rim_max_gaps_m": [p["arc_gap_m"]["max"] for p in loops],
        "edge_max_gap_m": max(
            [p["arc_gap_m"]["max"] for p in d["feature_diagnostics"]["edges"]],
            default=None,
        ),
        "support_early_m": d["support_error"]["early_contact_m"],
        "support_late_m": d["support_error"]["late_contact_m"],
        "generation_time_s": d["generation_time_s"],
        "review_required": d["review_required"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--caps", nargs="+", type=int, default=[32, 64])
    args = parser.parse_args()
    freeze()
    summary = []
    for name in args.cases:
        mesh = case_mesh(name)
        hull = convex_hull(mesh)
        structure = analyze(hull)
        for cap in args.caps:
            config = {"mode": "convex_surface", "max_spheres": cap, "seed": 5}
            final = generate_convex_surface_spheres(mesh, config)
            count = len(final.spheres)
            variants = {
                "B_random": "random",
                "C_fps": "fps",
                "D_features": "features",
                "E_auto": "auto",
            }
            for label, variant in variants.items():
                result = (
                    final
                    if variant == "auto"
                    else generate_convex_surface_spheres(
                        mesh, config, _variant=variant, _count=count
                    )
                )
                assert len(result.spheres) == count
                assert (
                    result.diagnostics["hull_sha256"]
                    == final.diagnostics["hull_sha256"]
                )
                assert np.array_equal(result.spheres[:, 3], final.spheres[:, 3])
                destination = OUT / name / f"{label}_{cap}"
                dump(
                    destination / "result.json",
                    {
                        "config": config,
                        "spheres_m": result.spheres.tolist(),
                        "diagnostics": result.diagnostics,
                    },
                )
                write_stage(destination / "preview.usda", mesh, result)
                summary.append(row(name, cap, label, result))
            if cap == 64:
                dump(
                    OUT / name / "obstacle_approaches.json",
                    obstacle_approaches(hull, final.spheres, structure),
                )
            print(
                name,
                cap,
                count,
                final.diagnostics["sampling_strategy"],
                final.diagnostics["radius_m"]["median"],
                flush=True,
            )
        if name in {"box", "bowl", "mug"}:
            history = []
            for cap in (32, 64):
                for variant in ("A_legacy", "G_auto"):
                    path = (
                        output_root()
                        / "structure_v2/ablations"
                        / name
                        / f"{variant}_{cap}/result.json"
                    )
                    previous = json.loads(path.read_text())
                    spheres = np.asarray(previous["spheres_object_local_m"])
                    history.append(
                        {
                            "source": str(path),
                            "actual_count": len(spheres),
                            "radius_m": stats(spheres[:, 3]),
                            "scope": "historical material spheres re-evaluated on hull, not a same-radius/count ablation",
                            "hull_evaluation": evaluate(hull, spheres, structure),
                        }
                    )
            dump(OUT / name / "historical_reference.json", history)
    dump(OUT / "ablation_summary.json", summary)
    calibration = []
    for name in ("mug", "bowl", "box"):
        hull = convex_hull(case_mesh(name))
        length = float(np.sqrt(hull.area / (2 * np.pi)))
        calibration.append(
            {
                "case": name,
                "hull_area_length_m": length,
                "candidate_A_006L_cap12mm": min(0.06 * length, 0.012),
                "selected_B_shared_bounded_affine": min(
                    0.1 * length, 0.006 + 0.02 * length, 0.012
                ),
                "rationale": "A undersizes mug vs frozen 8.56mm median; B stays in common historical size range, independent of count/class",
            }
        )
    dump(OUT / "radius_calibration.json", calibration)
    failures = []
    for name, mesh in (
        ("empty", trimesh.Trimesh()),
        (
            "coplanar",
            trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                faces=[],
                process=False,
            ),
        ),
        ("near_degenerate", trimesh.creation.box(extents=[0.2, 0.2, 1e-9])),
    ):
        try:
            generate_convex_surface_spheres(mesh)
        except ValueError as error:
            failures.append(
                {
                    "case": name,
                    "status": "rejected",
                    "error": str(error),
                    "spheres_returned": False,
                }
            )
        else:
            raise AssertionError(f"Invalid geometry unexpectedly accepted: {name}")
    dump(OUT / "failure_cases.json", failures)
    # Explicit real-asset rigid-transform parity. Site matching is unordered.
    if "box" in args.cases and "rotated_box" in args.cases:
        a = generate_convex_surface_spheres(case_mesh("box"))
        b = generate_convex_surface_spheres(case_mesh("rotated_box"))
        t = trimesh.transformations.euler_matrix(0.43, -0.68, 0.9)
        t[:3, 3] = [0.4, -0.8, 0.3]
        back = (b.spheres[:, :3] - t[:3, 3]) @ t[:3, :3]
        dump(
            OUT / "rotation_parity.json",
            {
                "counts": [len(a.spheres), len(b.spheres)],
                "bidirectional_site_error_m": max(
                    float(cKDTree(a.spheres[:, :3]).query(back)[0].max()),
                    float(cKDTree(back).query(a.spheres[:, :3])[0].max()),
                ),
            },
        )
    freeze()


if __name__ == "__main__":
    main()
