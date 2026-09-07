"""Convex-surface preview; fitting in a worker, Isaac imports only after Kit starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COLORS = {
    "corner": (1.0, 0.12, 0.2),
    "edge": (0.1, 0.8, 1.0),
    "rim": (1.0, 0.65, 0.06),
    "face": (0.25, 0.9, 0.4),
    "generic": (0.25, 0.5, 1.0),
    "random": (0.75, 0.4, 0.9),
}


def add_hull_layers(stage, diagnostic, *, root="/World", spheres=None):
    import colorsys
    import numpy as np
    import trimesh
    from pxr import Gf, UsdGeom
    from .usd_scene import bind, material, wire_mesh

    hull = trimesh.Trimesh(
        diagnostic["hull_vertices_m"], diagnostic["hull_faces"], process=False
    )
    structure = diagnostic["hull_structure"]
    scale = structure["scale_m"]

    def mesh_prim(path, faces, color):
        prim = UsdGeom.Mesh.Define(stage, path)
        prim.CreatePointsAttr(hull.vertices.tolist())
        prim.CreateFaceVertexCountsAttr([3] * len(faces))
        prim.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
        prim.CreateSubdivisionSchemeAttr("none")
        prim.CreateDoubleSidedAttr(True)
        mat = material(stage, path + "Material", color, 1)
        bind(prim.GetPrim(), mat)

    def curve(path, points, color, width):
        prim = UsdGeom.BasisCurves.Define(stage, path)
        prim.CreateTypeAttr("linear")
        prim.CreateCurveVertexCountsAttr([len(points)])
        prim.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        prim.CreateWidthsAttr([width])
        prim.SetWidthsInterpolation("constant")
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    mesh_prim(root + "/HullSolid", hull.faces, (0.45, 0.65, 0.72))
    wire_mesh(stage, root + "/HullWire", hull, (0.05, 0.8, 0.8), scale * 0.0007)
    for group in ("Patches", "HullFeatures", "Proxy", "Sites"):
        UsdGeom.Xform.Define(stage, root + "/" + group)
    for patch in structure["patches"]:
        color = colorsys.hsv_to_rgb((patch["id"] * 0.618034) % 1, 0.6, 0.85)
        mesh_prim(
            root + f"/Patches/p{patch['id']}", hull.faces[patch["face_ids"]], color
        )
    for edge in structure["major_edges"]:
        curve(
            root + f"/HullFeatures/edge_{edge['id']}",
            edge["points_m"],
            (0.15, 0.85, 1.0),
            scale * 0.003,
        )
    for loop in structure["rim_loops"]:
        curve(
            root + f"/HullFeatures/rim_{loop['id']}",
            loop["points_m"],
            (1.0, 0.5, 0.02),
            scale * 0.005,
        )
    box = structure["box"]
    if box["enabled"]:
        axes, origin = np.array(box["axes_columns"]), np.array(box["origin_m"])
        for i, color in enumerate(((1, 0, 0), (0, 1, 0), (0, 0.3, 1))):
            curve(
                root + f"/Proxy/axis_{i}",
                [origin, origin + axes[:, i] * scale * 0.8],
                color,
                scale * 0.005,
            )
        from itertools import product

        low, high = np.array(box["low_m"]), np.array(box["high_m"])
        corners = np.array(
            [
                origin + axes @ np.where(index, high, low)
                for index in product((0, 1), repeat=3)
            ]
        )
        for i in range(8):
            for j in range(i):
                if (i ^ j).bit_count() == 1:
                    curve(
                        root + f"/Proxy/e{i}_{j}",
                        corners[[i, j]],
                        (0.9, 0.2, 0.8),
                        scale * 0.001,
                    )
    if spheres is not None:
        for i, sphere in enumerate(spheres):
            prim = UsdGeom.Sphere.Define(stage, root + f"/Sites/p{i}")
            prim.CreateRadiusAttr(scale * 0.007)
            prim.AddTranslateOp().Set(Gf.Vec3d(*sphere[:3]))
            mat = material(
                stage, root + f"/Sites/m{i}", COLORS[diagnostic["sphere_sources"][i]], 1
            )
            bind(prim.GetPrim(), mat)
    return hull


def write_stage(path, mesh, result):
    from pxr import Usd
    from .mesh import write_stage as base_stage
    from .usd_scene import bind, material

    base_stage(path, mesh, result.spheres)
    stage = Usd.Stage.Open(str(path))
    add_hull_layers(stage, result.diagnostics, spheres=result.spheres)
    for i, kind in enumerate(result.regions):
        mat = material(stage, f"/World/ConvexMaterials/{kind}", COLORS[kind], 1)
        prim = stage.GetPrimAtPath(f"/World/Spheres/sphere_{i:03d}")
        bind(prim, mat)
        prim.SetCustomDataByKey("samplingSource", kind)
    stage.GetRootLayer().Save()


def hidden_paths(display):
    return [
        "/World/" + name
        for name in {
            "overlay": ["HullSolid", "Patches", "Proxy", "Sites", "ReferenceWire"],
            "spheres": [
                "ReferenceMesh",
                "ReferenceWire",
                "HullSolid",
                "Patches",
                "Proxy",
                "Sites",
            ],
            "original": [
                "HullSolid",
                "HullWire",
                "Patches",
                "Proxy",
                "Sites",
                "Spheres",
                "HullFeatures",
            ],
            "hull": [
                "ReferenceMesh",
                "ReferenceWire",
                "Patches",
                "Proxy",
                "Sites",
                "Spheres",
                "HullFeatures",
            ],
            "features": [
                "ReferenceMesh",
                "ReferenceWire",
                "HullSolid",
                "HullWire",
                "Spheres",
            ],
            "sites": [
                "ReferenceMesh",
                "ReferenceWire",
                "HullSolid",
                "Patches",
                "Proxy",
                "Spheres",
            ],
        }[display]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=Path,
        required=True,
        help="Existing asset preset; only source is used",
    )
    parser.add_argument(
        "--config", type=Path, help="Optional convex_surface generation JSON"
    )
    parser.add_argument("--max-spheres", type=int)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--open-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--display",
        choices=("overlay", "spheres", "original", "hull", "features", "sites"),
        default="spheres",
    )
    parser.add_argument("--view", choices=("close", "top", "front"), default="close")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    library = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != library:
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=library + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        os.execvpe(
            sys.executable,
            [
                sys.executable,
                "-m",
                "asset_collision_spheres.preview.convex",
                *sys.argv[1:],
            ],
            env,
        )
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    config = (
        json.loads(args.config.read_text())
        if args.config
        else {"mode": "convex_surface"}
    )
    if args.max_spheres is not None:
        config["max_spheres"] = args.max_spheres
    path = args.output.resolve() / "convex_surface.usda"
    if not args.build_only and not args.prepare_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "asset_collision_spheres.preview.convex",
                *sys.argv[1:],
                "--prepare-only",
            ],
            check=True,
        )
    else:
        from ..algorithms.convex_surface import generate_convex_surface_spheres
        from ..geometry.validation import mesh_fingerprint
        from ..loaders.presets import read_preset
        from ..loaders.usd import load_material_mesh

        preset, _ = read_preset(args.preset)
        mesh, source = load_material_mesh(preset, require_material=False)
        if args.open_existing:
            report = json.loads(path.with_suffix(".json").read_text())
            if (
                report["config"] != config
                or report["source"] != source
                or report["original_mesh_sha256"] != mesh_fingerprint(mesh)
                or report["usd_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "Saved convex preview changed; rebuild in a separate output"
                )
        else:
            result = generate_convex_surface_spheres(mesh, config)
            args.output.mkdir(parents=True, exist_ok=True)
            write_stage(path, mesh, result)
            report = {
                "schema": "asset-collision-spheres/convex-preview/1",
                "config": config,
                "source": source,
                "original_mesh_sha256": mesh_fingerprint(mesh),
                "spheres_m": result.spheres.tolist(),
                "diagnostics": result.diagnostics,
                "usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "native_attachment_verified": False,
            }
            path.with_suffix(".json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "usd": str(path),
                        "count": len(result.spheres),
                        "strategy": result.diagnostics["sampling_strategy"],
                    }
                ),
                flush=True,
            )
    if not args.build_only and not args.prepare_only:
        from .isaac import open_isaac

        args.camera_path = (
            "/World/Camera"
            + {"front": "Side", "top": "Top", "close": "Close"}[args.view]
        )
        args.hidden_paths = hidden_paths(args.display)
        args.preview_legend = "CONVEX V3 | cyan=hull; red=corner; cyan=edge; gold=rim; green=face; blue=generic | static diagnostic"
        open_isaac(path, args)


if __name__ == "__main__":
    main()
