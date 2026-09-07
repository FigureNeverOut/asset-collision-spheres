"""Generate collision spheres from a mesh or an explicit USD preset."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .auto_geometry_spheres import generate_auto_geometry_spheres
from .io import load_mesh, make_report
from .mesh_region_spheres import generate_mesh_region_spheres


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mesh", type=Path, help="Single OBJ/PLY/STL material mesh")
    source.add_argument("--preset", type=Path, help="JSON containing source and generation (USD)")
    parser.add_argument("--unit-scale-m", type=float, help="Required for --mesh: metres per input unit")
    parser.add_argument("--config", type=Path, help="Required for --mesh: generation JSON")
    parser.add_argument("--budget", type=int, help="Override sphere_budget")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    args = parser.parse_args(argv)
    if args.mesh and (args.config is None or args.unit_scale_m is None):
        parser.error("--mesh requires --config and --unit-scale-m")
    if args.preset and (args.config is not None or args.unit_scale_m is not None):
        parser.error("--preset includes its own generation config and units")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    try:
        source_info = None
        frame = "input_mesh_axes_m"
        if args.preset:
            from .usd import load_material_mesh

            preset = deepcopy(json.loads(args.preset.read_text()))
            asset = Path(preset["source"]["usd_path"]).expanduser()
            if not asset.is_absolute():
                asset = args.preset.resolve().parent / asset
            preset["source"]["usd_path"] = str(asset)
            mesh, source_info = load_material_mesh(preset)
            config = dict(preset["generation"])
            frame = "usd_stage_transformed_z_up_m"
        else:
            mesh = load_mesh(args.mesh, unit_scale_m=args.unit_scale_m)
            config = dict(json.loads(args.config.read_text()))
        if args.budget is not None:
            config["sphere_budget"] = args.budget
        generate = generate_mesh_region_spheres if "regions" in config else generate_auto_geometry_spheres
        result = generate(mesh, config)
        report = make_report(mesh, result, config, coordinate_frame=frame)
        if source_info is not None:
            # Keep source transforms/dimensions, without recording machine paths.
            source_info.pop("usd_path", None)
            report["source"] = source_info
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(serialized)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        parser.exit(2, f"asset-spheres: {error}\n")
    print(f"Wrote {len(result.spheres)} spheres to {args.output}")
    if result.diagnostics.get("review_required"):
        print("Review required: some material components have no sphere representation.")
    return 0
