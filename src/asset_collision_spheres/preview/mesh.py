"""Standalone full-material USD mesh sphere preview: no robot or physics.

Explicit asset units/mesh selection belong to the loader, not the fitter.
Annotation labels, arrows and object categories never enter generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .usd_scene import bind, material, set_matrix, wire_mesh
from .isaac import open_isaac
from ..paths import config_path, output_root, workspace_root
from ..loaders.presets import read_preset

def load_material_mesh(preset):
    from ..loaders.usd import load_material_mesh as load
    return load(preset)

ROOT = workspace_root(required=False) or Path.cwd()
DEFAULT_PRESET = config_path("bowl/bowl_001_auto_geometry.json")






def write_stage(path, mesh, spheres, features=None):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    gray = material(stage, "/World/Looks/Reference", (0.72, 0.75, 0.78), 1)
    orange = material(stage, "/World/Looks/Sphere", (1.0, 0.52, 0.08), 1)
    reference = UsdGeom.Mesh.Define(stage, "/World/ReferenceMesh")
    reference.CreatePointsAttr(mesh.vertices.tolist())
    reference.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    reference.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    reference.CreateSubdivisionSchemeAttr("none")
    reference.CreateDoubleSidedAttr(True)
    bind(reference.GetPrim(), gray)
    # A typed Imageable parent is necessary for --display mesh to hide all balls.
    UsdGeom.Xform.Define(stage, "/World/Spheres")
    for index, (x, y, z, radius) in enumerate(spheres):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Spheres/sphere_{index:03d}")
        sphere.CreateRadiusAttr(float(radius))
        sphere.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        bind(sphere.GetPrim(), orange)
    size = float(max(mesh.extents))
    # Same coordinates as the gray mesh and spheres; no separate displaced ghost.
    wire_mesh(stage, "/World/ReferenceWire", mesh, (0.65, 0.25, 0.8), size * 0.00045)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(900)
    if features is not None:
        from .features import add_features
        add_features(stage, mesh, features)
    target = mesh.bounds.mean(axis=0)
    # Frame tall boxes as well as shallow bowls. At 32 mm focal length, the
    # 1280x900 viewport's vertical half-angle is about 13 degrees. Bound the full
    # object by a sphere so rotated corners cannot be cropped by the close view.
    frame_distance = float(np.linalg.norm(mesh.extents)) * 0.5 / 0.225 * 1.12
    for name, offset, up in (
        ("Close", (1.1, -1.6, 1.35), (0, 0, 1)),
        ("Top", (0, 0, 3.0), (0, 1, 0)),
        ("Side", (0, -2.4, 0.2), (0, 0, 1)),
    ):
        camera = UsdGeom.Camera.Define(stage, "/World/Camera" + name)
        direction = np.asarray(offset, dtype=float)
        eye = target + direction / np.linalg.norm(direction) * frame_distance
        set_matrix(
            camera.GetPrim(),
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
            .GetInverse(),
        )
        camera.CreateFocalLengthAttr(32)
        camera.CreateClippingRangeAttr(Gf.Vec2f(max(size * 0.001, 1e-5), size * 100))
    stage.GetRootLayer().Save()


def artifact_stem(config):
    # Keep preview startup free of NumPy/USD before spawning the fitting worker.
    extended = any(k in config for k in (
        "algorithm_variant", "shape_hint", "sphere_count_mode", "outward_offset_mode", "radius_mode",
        "feature_mode", "max_spheres", "backend_sphere_capacity", "extra_face_spheres", "system_policy",
    ))
    if extended:
        adaptive = config.get("sphere_count_mode", "fixed" if "sphere_budget" in config else "adaptive") == "adaptive"
        return "joint_adaptive_" + str(config.get("max_spheres") or "auto") if adaptive else "joint_fixed_" + str(config["sphere_budget"])
    return "auto_geometry_" + str(config["sphere_budget"])


def build(preset, config, output):
    from asset_collision_spheres.algorithms.auto_geometry import (
        generate_auto_geometry_spheres,
    )
    from asset_collision_spheres.adapters.fastsim import (
        check_spheres,
        mesh_fingerprint,
    )

    mesh, source = load_material_mesh(preset)
    print(
        f"[fit] {len(mesh.faces)} triangles; budget={config.get('sphere_budget', 'adaptive')}; no labels",
        flush=True,
    )
    result = generate_auto_geometry_spheres(mesh, config)
    check_spheres(
        mesh, result.spheres, result.diagnostics.get("effective_hard_limit", config.get("sphere_budget", 256)), result.diagnostics["max_outward_offset_m"]
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / (artifact_stem(config) + ".usda")
    write_stage(path, mesh, result.spheres, result.diagnostics.get("feature_diagnostics"))
    report = {
        "schema": "fastsim/static-material-sphere-preview/1",
        "source": source,
        "mesh_sha256": mesh_fingerprint(mesh),
        "generation_config": config,
        "spheres_preview_local_m": result.spheres.tolist(),
        "fit": result.diagnostics,
        "robot_attachment_tested": False,
        "physics_task_executed": False,
        "usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "usd": str(path),
                "actual_count": len(result.spheres),
                "diagnostics": result.diagnostics["diagnostics"],
            }
        ),
        flush=True,
    )
    return path


def verify_existing(path, preset, config):
    report = json.loads(path.with_suffix(".json").read_text())
    if report["generation_config"] != config:
        raise ValueError("Saved configuration differs; rebuild without --open-existing")
    mesh, source = load_material_mesh(preset)
    from asset_collision_spheres.adapters.fastsim import (
        mesh_fingerprint,
    )

    if source != report["source"] or mesh_fingerprint(mesh) != report["mesh_sha256"]:
        raise ValueError("Source geometry changed; rebuild without --open-existing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["usd_sha256"]:
        raise ValueError("Saved preview USD changed; rebuild without --open-existing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--budget", type=int, help="Override only sphere_budget")
    parser.add_argument("--config", type=Path, help="Generation-only override, leaving source preset untouched")
    parser.add_argument(
        "--output", type=Path, help="Separate generated-output directory"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--open-existing", action="store_true")
    parser.add_argument("--view", choices=("close", "top", "side"), default="close")
    parser.add_argument(
        "--display", choices=("overlay", "spheres", "mesh", "features"), default="overlay"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    library = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != library:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = library + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        os.execvpe(sys.executable, [sys.executable, "-m", "asset_collision_spheres.preview.mesh", *sys.argv[1:]], env)
    preset, config = read_preset(args.preset, args.budget)
    if args.config:
        config = json.loads(args.config.read_text())
        if args.budget is not None:
            config["sphere_budget"] = args.budget
    if args.budget is not None and config.get("sphere_count_mode") == "adaptive":
        parser.error("--budget is for fixed mode; adaptive mode uses max_spheres in --config")
    output = (
        args.output.resolve()
        if args.output
        else output_root() / ("structure_v2" if artifact_stem(config).startswith("joint_") else "") / args.preset.stem / (args.config.stem if args.config else "")
    )
    path = output / (artifact_stem(config) + ".usda")
    if not args.build_only and not args.prepare_only:
        # Standalone pxr/trimesh/backend imports must never precede SimulationApp.
        # Use a separate process even for the saved-geometry validation path.
        subprocess.run([sys.executable, "-m", "asset_collision_spheres.preview.mesh", *sys.argv[1:], "--prepare-only"], check=True)
    elif args.open_existing:
        verify_existing(path, preset, config)
    else:
        path = build(preset, config, output)
    if not args.build_only and not args.prepare_only:
        args.camera_path = "/World/Camera" + args.view.title()
        args.hidden_paths = {
            "overlay": ["/World/ReferenceWire"],
            "spheres": ["/World/ReferenceMesh"],
            "mesh": ["/World/Spheres", "/World/ReferenceWire"],
            "features": ["/World/Spheres", "/World/ReferenceWire", "/World/ReferenceMesh"],
        }[args.display]
        if args.display != "features":
            args.hidden_paths.append("/World/Features")
        args.preview_legend = "STATIC MESH PREVIEW | orange=spheres; gray=reference; purple=co-located reference wire; no robot/physics"
        open_isaac(path, args)


if __name__ == "__main__":
    main()
