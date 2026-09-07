"""Opt-in v4 routing/sampling regression, preserving all earlier artifacts."""

import argparse
import json
import time

import numpy as np
from freeze_geometry_reference import freeze
from geometry_cases import synthetic_case
from run_convex_experiments import case_mesh

from asset_collision_spheres.api import (
    generate_convex_surface_spheres,
    generate_geometry_spheres,
)
from asset_collision_spheres.paths import output_root
from asset_collision_spheres.preview.geometry import write_stage

POSITIVE = ["mug", "bowl", "box", "tray"]
NEGATIVE = [
    "solid_box",
    "sphere",
    "straight_rod",
    "curved_rod",
    "u_solid",
    "l_solid",
    "ring_solid",
    "branched_solid",
]
BOUNDARY = ["hollow_bent_tube", "narrow_neck", "irregular_scan", "rounded_box"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--cases", nargs="+", default=POSITIVE + NEGATIVE + BOUNDARY)
    parser.add_argument(
        "--output",
        type=__import__("pathlib").Path,
        default=output_root() / "geometry_v4/regression_01",
    )
    args = parser.parse_args()
    freeze()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    started = time.perf_counter()
    for name in args.cases:
        mesh = case_mesh(name) if name in POSITIVE else synthetic_case(name)
        for cap in args.caps:
            config = {
                "mode": "auto_geometry",
                "geometry_policy": "auto",
                "max_spheres": cap,
                "seed": 5,
            }
            result = generate_geometry_spheres(mesh, config)
            d = result.diagnostics
            routing = d["geometry_routing"]
            expected = "convex_hull" if name in POSITIVE else "original_surface"
            row = {
                "case": name,
                "max_spheres": cap,
                "actual_count": len(result.spheres),
                "selected": d["geometry_policy_selected"],
                "expected": expected,
                "routing_matches_expectation": d["geometry_policy_selected"]
                == expected,
                "confidence": routing["container_envelope_confidence"],
                "review_required": d["review_required"],
                "strategy": d["sampling_strategy"],
                "radius_m": d["radius_m"],
                "surface_site_max_error_m": d["surface_site_max_error_m"],
                "site_distance_m": d["surface_uniformity"]["site_distance_m"],
                "sphere_surface_gap_m": d["surface_uniformity"]["sphere_surface_gap_m"],
                "generation_time_s": d["generation_time_s"],
            }
            if d["geometry_policy_selected"] == "convex_hull":
                legacy = generate_convex_surface_spheres(
                    mesh, {"max_spheres": cap, "seed": 5}
                )
                row["same_as_legacy_v3_sphere_array"] = bool(
                    np.array_equal(result.spheres, legacy.spheres)
                )
            destination = args.output / name / str(cap)
            destination.mkdir(parents=True)
            (destination / "result.json").write_text(
                json.dumps(
                    {
                        "config": config,
                        "spheres_m": result.spheres.tolist(),
                        "diagnostics": d,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            write_stage(destination / "preview.usda", mesh, result)
            summary.append(row)
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(row), flush=True)
    freeze()
    print(
        f"Completed {len(summary)} runs in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if not all(r["routing_matches_expectation"] for r in summary):
        raise SystemExit(
            "Some routing expectations failed; inspect summary, do not relabel"
        )


if __name__ == "__main__":
    main()
