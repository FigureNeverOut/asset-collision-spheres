"""Frozen 32-budget mug/bowl comparison, native public MORPHIT versus auto geometry.

Copies the approved static stage; changes only its displayed spheres. No robot
planning/attachment operation is executed, and source previews remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from ...paths import workspace_root, output_root, config_path

from . import mug
from ...preview import mesh as standalone
from .book import bind, material, open_isaac, set_matrix

ROOT = workspace_root()
LEGACY_OUTPUT = ROOT / "task2sim/runs/collision_ball_test/curobo_matched_32"
OUTPUT = output_root() / "comparisons/curobo_matched_32"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(asset):
    import numpy as np
    from asset_collision_spheres.adapters.fastsim import (
        mesh_fingerprint,
    )
    from pxr import Usd

    if asset == "mug":
        path = ROOT / "task2sim/runs/1e/mug_auto_geometry_32/auto_geometry.usda"
        saved = json.loads(path.with_suffix(".json").read_text())
        mesh = mug.mesh_data(Usd.Stage.Open(saved["assets"]["mug"]), True)
        geometry_config_path = (
            config_path("mug/1e_mug_auto_geometry_32.json")
        )
        config = json.loads(geometry_config_path.read_text())
        spheres = saved["spheres_object_local_m"]
    else:
        preset, config = standalone.read_preset(standalone.DEFAULT_PRESET, 32)
        path = (
            ROOT
            / "task2sim/runs/collision_ball_test/bowl_001_auto_geometry/auto_geometry_32.usda"
        )
        standalone.verify_existing(path, preset, config)
        saved = json.loads(path.with_suffix(".json").read_text())
        mesh, _ = standalone.load_material_mesh(preset)
        spheres = saved["spheres_preview_local_m"]
    fingerprint = mesh_fingerprint(mesh)
    if saved["fit"]["mesh_sha256"] != fingerprint:
        raise ValueError("Saved improved spheres do not belong to this input mesh")
    if saved["fit"]["config"]["sphere_budget"] != 32 or len(spheres) > 32:
        raise ValueError("Frozen comparison requires the approved 32-budget result")
    if any(saved["fit"]["config"].get(k) != v for k, v in config.items()):
        raise ValueError("Improved config differs from the saved generated result")
    return path, mesh, np.asarray(spheres), config, fingerprint


def evaluate(mesh, spheres, config):
    """Common held-out probes, without imposing the new fitter's bounds on MORPHIT."""
    import trimesh
    from asset_collision_spheres.algorithms.auto_geometry import (
        free_samples,
    )
    from asset_collision_spheres.algorithms.mesh_regions import (
        material_depth,
        sphere_clearance,
    )

    surface, faces = trimesh.sample.sample_surface(mesh, 12000, seed=20260908)
    radius = config["probe_radius_m"]
    witnesses = surface + mesh.face_normals[faces] * (
        radius - config["probe_penetration_m"]
    )
    reference_hit = material_depth(mesh, witnesses) >= -radius
    detected = sphere_clearance(witnesses, spheres, radius) <= 0
    points, _, beyond, free_radius = free_samples(
        mesh, surface, faces, config, seed=20260909
    )
    intrusion = sphere_clearance(points, spheres, free_radius) < 0
    return {
        "seed": 20260908,
        "contact_probe_count": int(reference_hit.sum()),
        "contact_detection_fraction": float(detected[reference_hit].mean()),
        "surface_coverage_fraction": float(
            (sphere_clearance(surface, spheres) <= 1e-8).mean()
        ),
        "free_probe_count_beyond_new_allowance": int(beyond.sum()),
        "free_probe_intrusions_beyond_new_allowance": int((intrusion & beyond).sum()),
        "note": "Identical finite probes; not continuous/task collision certification. 6 mm band is an evaluation threshold, not a baseline fitting constraint.",
    }


def replace_spheres(source, target, spheres, asset):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    stage.GetRootLayer().Export(str(target))
    stage = Usd.Stage.Open(str(target))
    root_path = "/World/CollisionSpheres" if asset == "mug" else "/World/Spheres"
    frame = (
        UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath("/World/MugAtAttach")
        )
        if asset == "mug"
        else Gf.Matrix4d(1)
    )
    stage.RemovePrim(root_path)
    root = UsdGeom.Xform.Define(stage, root_path)
    set_matrix(root.GetPrim(), frame)
    mat = material(stage, "/World/MatchedSphereMaterial", (1.0, 0.52, 0.08), 1)
    for i, (x, y, z, radius) in enumerate(spheres):
        sphere = UsdGeom.Sphere.Define(stage, f"{root_path}/sphere_{i:03d}")
        sphere.CreateRadiusAttr(float(radius))
        sphere.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        bind(sphere.GetPrim(), mat)
    stage.GetRootLayer().Save()


def build(asset):
    source, mesh, improved, config, fingerprint = inputs(asset)
    out = OUTPUT / asset
    out.mkdir(parents=True, exist_ok=True)
    before = digest(source)
    print(
        f"[fit] {asset}: public cuRobo MORPHIT; maximum 32; CUDA; seed={config['seed']}",
        flush=True,
    )
    baseline, fit = mug.fit_baseline(mesh, 32, "cuda", config["seed"])
    if not 1 <= len(baseline) <= 32:
        raise ValueError(
            "cuRobo output exceeds maximum 32 or is empty; no silent truncation"
        )
    report = {
        "asset": asset,
        "source_stage": str(source),
        "source_stage_sha256": before,
        "source_report_sha256": digest(source.with_suffix(".json")),
        "mesh_sha256": fingerprint,
        "sphere_budget": 32,
        "seed": config["seed"],
        "new_config": config,
        "baseline_fit": fit,
        "comparison": "Same full-material mesh, scale, seed, maximum budget, frozen pose/cameras and sphere display material. Different algorithm sampling/objectives; baseline has no 6 mm outward constraint. Native fallback retained; no padding/truncation.",
        "native_attachment_or_physics_tested_this_run": False,
        "methods": {},
    }
    for method, spheres in (("baseline", baseline), ("auto_geometry", improved)):
        path = out / f"{method}_32.usda"
        replace_spheres(source, path, spheres, asset)
        report["methods"][method] = {
            "actual_count": len(spheres),
            "spheres_object_local_m": [list(map(float, s)) for s in spheres],
            "usd_sha256": digest(path),
            "evaluation": evaluate(mesh, spheres, config),
        }
    if digest(source) != before:
        raise RuntimeError("Source preview changed during comparison")
    (out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "baseline_actual": len(baseline),
                "improved_actual": len(improved),
                "native_debug": fit["debug_info"],
            },
            default=str,
        ),
        flush=True,
    )


def verify(asset, method):
    path = OUTPUT / asset / f"{method}_32.usda"
    report = json.loads((path.parent / "comparison.json").read_text())
    source = Path(report["source_stage"])
    if (
        digest(source) != report["source_stage_sha256"]
        or digest(source.with_suffix(".json")) != report["source_report_sha256"]
    ):
        raise ValueError("Source preview changed; rebuild comparison")
    if digest(path) != report["methods"][method]["usd_sha256"]:
        raise ValueError("Comparison USD changed; rebuild comparison")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=("mug", "bowl"), required=True)
    parser.add_argument(
        "--method", choices=("baseline", "auto_geometry"), default="baseline"
    )
    parser.add_argument(
        "--budget",
        type=int,
        choices=(32,),
        default=32,
        help="Matched to the frozen approved 32-budget previews",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--open-existing", action="store_true")
    parser.add_argument("--view", choices=("close", "top", "side"), default="close")
    parser.add_argument("--display", choices=("overlay", "spheres"), default="overlay")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    lib = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != lib:
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        os.execvpe(sys.executable, [sys.executable, "-m", "asset_collision_spheres.adapters.workspace.curobo_matched", *sys.argv[1:]], env)
    if args.build_only or args.prepare_only:
        if not args.open_existing:
            build(args.asset)
        verify(args.asset, args.method)
        return
    # Keep pxr/CUDA fitter imports out of the Kit process; bound fitter failures.
    subprocess.run(
        [sys.executable, "-m", "asset_collision_spheres.adapters.workspace.curobo_matched", *sys.argv[1:], "--prepare-only"], check=True, timeout=600
    )
    path = verify(args.asset, args.method)
    args.camera_path = "/World/Camera" + args.view.title()
    if args.asset == "mug":
        args.hidden_paths = ["/World/MugAtAttach"] if args.display == "spheres" else []
    else:
        args.hidden_paths = (
            ["/World/ReferenceMesh"]
            if args.display == "spheres"
            else ["/World/ReferenceWire"]
        )
    args.preview_legend = f"{args.asset} {args.method} | MAXIMUM 32 (actual count in comparison.json); same mesh/pose/camera; static only"
    open_isaac(path, args)


if __name__ == "__main__":
    main()
