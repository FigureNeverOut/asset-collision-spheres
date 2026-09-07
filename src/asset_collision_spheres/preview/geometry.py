"""V4 target surface and routing evidence layers (standalone mesh, no physics)."""

from __future__ import annotations


def write_stage(path, mesh, result):
    import numpy as np
    import trimesh
    from pxr import Gf, Usd, UsdGeom

    from ..algorithms.original_surface import surface_mesh
    from .convex import COLORS
    from .mesh import write_stage as base_stage
    from .usd_scene import bind, material, wire_mesh

    base_stage(path, mesh, result.spheres)
    stage = Usd.Stage.Open(str(path))
    diagnostic = result.diagnostics
    routing = diagnostic["geometry_routing"]
    convex = routing["geometry_policy_selected"] == "convex_hull"
    target = (
        trimesh.Trimesh(
            diagnostic["hull_vertices_m"], diagnostic["hull_faces"], process=False
        )
        if convex
        else surface_mesh(mesh)
    )
    structure = diagnostic["hull_structure" if convex else "surface_structure"]
    scale = structure["scale_m"]
    wire_mesh(
        stage,
        "/World/TargetWire",
        target,
        (0.05, 0.8, 0.8) if convex else (0.65, 0.25, 0.8),
        scale * 0.001,
    )
    stage.GetPrimAtPath("/World").SetCustomDataByKey(
        "geometryPolicySelected", routing["geometry_policy_selected"]
    )
    UsdGeom.Xform.Define(stage, "/World/Features")
    UsdGeom.Xform.Define(stage, "/World/RoutingEvidence")

    def line(path, points, color, width):
        prim = UsdGeom.BasisCurves.Define(stage, path)
        prim.CreateTypeAttr("linear")
        prim.CreateCurveVertexCountsAttr([len(points)])
        prim.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        prim.CreateWidthsAttr([width])
        prim.SetWidthsInterpolation("constant")
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    for kind, features in [
        ("edge", structure["major_edges"]),
        ("rim", structure["rim_loops"]),
    ]:
        for i, feature in enumerate(features):
            line(
                f"/World/Features/{kind}{i}",
                feature["points_m"],
                COLORS[kind],
                scale * 0.004,
            )
    if "analysis_hull_vertices_m" in routing:
        hull = trimesh.Trimesh(
            routing["analysis_hull_vertices_m"],
            routing["analysis_hull_faces"],
            process=False,
        )
        wire_mesh(
            stage,
            "/World/RoutingEvidence/AnalysisHull",
            hull,
            (0.9, 0.45, 0.15),
            scale * 0.001,
        )
    for i, cap in enumerate(routing.get("candidate_caps", [])):
        center, normal = np.asarray(cap["center_m"]), np.asarray(cap["normal"])
        depth = cap["center_free_depth_m"] or cap["axis_extent_m"] * 0.3
        line(
            f"/World/RoutingEvidence/cap{i}",
            [center, center - normal * depth],
            (0, 1, 0) if cap["eligible"] else (1, 0, 0),
            scale * 0.006,
        )
        for j, section in enumerate(cap["sections"]):
            segments = trimesh.intersections.mesh_plane(
                mesh, normal, section["origin_m"]
            )
            for k, segment in enumerate(segments):
                line(
                    f"/World/RoutingEvidence/section{i}_{j}/s{k}",
                    segment,
                    (1, 0.6, 0),
                    scale * 0.003,
                )
    for i, kind in enumerate(result.regions):
        bind(
            stage.GetPrimAtPath(f"/World/Spheres/sphere_{i:03d}"),
            material(stage, f"/World/GeometryMaterials/{kind}", COLORS[kind], 1),
        )
    # Save clean default view; the inspector uses a session layer to enable
    # analysis explicitly. An analysis hull is NEVER presented as the target.
    for name in ("Features", "RoutingEvidence", "ReferenceMesh", "ReferenceWire"):
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/" + name)).MakeInvisible()
    stage.GetRootLayer().Save()


def main():
    import argparse
    import json
    from pathlib import Path

    from ..api import generate_geometry_spheres, load_mesh
    from ..loaders.usd import load_material_mesh

    parser = argparse.ArgumentParser(
        description="Generate V4 JSON/USD offline; use preview.inspect to open Isaac"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--preset",
        type=Path,
        help="Existing source preset; old generation section is not used",
    )
    source.add_argument("--mesh", type=Path)
    parser.add_argument("--unit-scale-m", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-spheres", type=int)
    parser.add_argument(
        "--geometry-policy", choices=("auto", "convex_hull", "original_surface")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = (
        json.loads(args.config.read_text())
        if args.config
        else {"geometry_policy": "auto"}
    )
    if args.max_spheres is not None:
        config["max_spheres"] = args.max_spheres
    if args.geometry_policy is not None:
        config["geometry_policy"] = args.geometry_policy
    if args.output.exists():
        parser.error("Choose a new output directory; old results are never overwritten")
    if args.preset:
        preset = json.loads(args.preset.read_text())
        asset = Path(preset["source"]["usd_path"])
        if not asset.is_absolute():
            preset["source"]["usd_path"] = str(args.preset.resolve().parent / asset)
        mesh, source_info = load_material_mesh(
            preset, require_material=False, require_triangles=True
        )
    else:
        if args.unit_scale_m is None:
            parser.error("--mesh requires --unit-scale-m")
        mesh = load_mesh(
            args.mesh, unit_scale_m=args.unit_scale_m, require_material=False
        )
        source_info = {
            "mesh_path": str(args.mesh.resolve()),
            "unit_scale_m": args.unit_scale_m,
        }
    result = generate_geometry_spheres(mesh, config)
    args.output.mkdir(parents=True)
    write_stage(args.output / "preview.usda", mesh, result)
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "config": config,
                "source": source_info,
                "spheres_m": result.spheres.tolist(),
                "diagnostics": result.diagnostics,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        f"{result.diagnostics['geometry_policy_selected']}: {len(result.spheres)} spheres; {args.output / 'preview.usda'}"
    )


if __name__ == "__main__":
    main()
