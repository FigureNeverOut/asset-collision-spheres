"""Generate collision spheres from a mesh or an explicit USD preset."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .api import (
    generate_auto_geometry_spheres,
    generate_mesh_region_spheres,
    generate_convex_surface_spheres,
    load_mesh,
    make_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    from . import __version__

    parser.add_argument("--version", action="version", version=__version__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mesh", type=Path, help="Single OBJ/PLY/STL material mesh")
    source.add_argument(
        "--preset", type=Path, help="JSON containing source and generation (USD)"
    )
    parser.add_argument(
        "--unit-scale-m", type=float, help="Required for --mesh: metres per input unit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional generation JSON; without it --mesh defaults to v4 auto",
    )
    parser.add_argument(
        "--budget", type=int, help="Legacy only: override sphere_budget"
    )
    parser.add_argument(
        "--geometry-policy", choices=("auto", "convex_hull", "original_surface")
    )
    parser.add_argument(
        "--max-spheres", type=int, help="Surface modes: maximum count, default 64"
    )
    parser.add_argument("--shape-hint", choices=("auto", "box", "open_box", "generic"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    args = parser.parse_args(argv)
    if args.mesh and args.unit_scale_m is None:
        parser.error("--mesh requires --unit-scale-m")
    if args.preset and (args.config is not None or args.unit_scale_m is not None):
        parser.error("--preset includes its own generation config and units")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")

    def overrides(config):
        result = dict(config)
        for key in ("geometry_policy", "max_spheres", "shape_hint", "seed"):
            value = getattr(args, key)
            if value is not None:
                result[key] = value
        return result

    try:
        source_info = None
        frame = "input_mesh_axes_m"
        if args.preset:
            from .loaders.usd import load_material_mesh

            preset = deepcopy(json.loads(args.preset.read_text()))
            asset = Path(preset["source"]["usd_path"]).expanduser()
            if not asset.is_absolute():
                asset = args.preset.resolve().parent / asset
            preset["source"]["usd_path"] = str(asset)
            config = overrides(
                preset.get(
                    "generation", {"mode": "auto_geometry", "geometry_policy": "auto"}
                )
            )
            mesh, source_info = load_material_mesh(
                preset,
                require_material=config.get("mode") != "convex_surface"
                and "geometry_policy" not in config,
                require_triangles="geometry_policy" in config,
            )
            frame = "usd_stage_transformed_z_up_m"
        else:
            config = overrides(
                json.loads(args.config.read_text())
                if args.config
                else {"mode": "auto_geometry", "geometry_policy": "auto"}
            )
            mesh = load_mesh(
                args.mesh,
                unit_scale_m=args.unit_scale_m,
                require_material=config.get("mode") != "convex_surface"
                and "geometry_policy" not in config,
            )
        if args.budget is not None:
            if config.get("mode") == "convex_surface" or "geometry_policy" in config:
                raise ValueError(
                    "convex_surface uses max_spheres in its configuration, not --budget"
                )
            if config.get("sphere_count_mode") == "adaptive":
                raise ValueError(
                    "--budget is for fixed mode; set max_spheres for adaptive mode"
                )
            config["sphere_budget"] = args.budget
        generate = (
            generate_mesh_region_spheres
            if "regions" in config
            else generate_auto_geometry_spheres
        )
        if config.get("mode") == "convex_surface":
            generate = generate_convex_surface_spheres
        result = generate(mesh, config)
        report = make_report(mesh, result, config, coordinate_frame=frame)
        if source_info is not None:
            # Keep source transforms/dimensions, without recording machine paths.
            source_info.pop("usd_path", None)
            report["source"] = source_info
        serialized = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(serialized)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        parser.exit(2, f"asset-spheres: {error}\n")
    print(f"Wrote {len(result.spheres)} spheres to {args.output}")
    if result.diagnostics.get("review_required"):
        print(
            "Review required: inspect missing material/structure and resource limits in diagnostics."
        )
    return 0
