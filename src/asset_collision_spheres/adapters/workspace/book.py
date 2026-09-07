"""Static Isaac preview, not a physics replay or native attachment attestation.

Uses the current 1g assets and a recorded physical-attach book pose. Robot links
are reconstructed from the preceding successful joint target plus gripper close;
they are not a synchronized measured articulation snapshot. Both fitters consume
the SAME actual collision mesh, including any visual/collision offset in it.
Only generated preview layers are authored; source assets/configs stay untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from ...paths import workspace_root, output_root, config_path
from ...preview.usd_scene import matrix, set_matrix, mesh_data, pose_robot, material, bind, wire_mesh
from ...preview.isaac import open_isaac

ROOT = workspace_root()
PROJECT = (
    ROOT / "task2sim/runs/1g/scene028/inst_0/executions/20260904_165903/fastsim_project"
)
LEGACY_OUTPUT = PROJECT / "sphere_preview"
DEFAULT_OUTPUT = output_root() / "book/sphere_preview"








def source_state():
    import yaml

    run = yaml.safe_load((PROJECT / "run.yaml").read_text())
    log = PROJECT / "diagnostics/visible_yaml_only_retry28.log"
    traces = [
        json.loads(line.split(" ", 1)[1])
        for line in log.read_text().splitlines()
        if line.startswith("FASTSIM_ATTACHMENT_TRANSFORM_TRACE ")
    ]
    trace = next(
        t
        for t in traces
        if t["object_alias"] == "book_a" and t["operation"] == "attach"
    )
    record = trace["records"][0]
    frames = {
        t["source_frame"]: t["target_T_source"] for t in trace["endpoint_transforms"]
    }
    outcomes = PROJECT / "diagnostics/visible_yaml_only_retry28_arm_outcomes.jsonl"
    rows = [
        json.loads(line) for line in outcomes.read_text().splitlines() if line.strip()
    ]
    outcome = next(r for r in rows if r.get("selected_pose_candidate") == "grasp_44")
    robot = run["scenario"]["scene"]["robots"]["g2"]
    joints = {
        k: float(v["position"]) for k, v in robot["initial_state"]["joints"].items()
    }
    joints.update(zip(outcome["joint_ids"], outcome["positions"], strict=True))
    joints["idx31_gripper_l_inner_joint1"] = -0.7

    def asset(name):
        component = yaml.safe_load((PROJECT / "components" / name).read_text())
        return Path(component["variants"]["isaaclab"]["resources"]["model"]["uri"])

    return {
        "run": run,
        "robot_asset": asset("g2-crsb-swiftpicker.yaml"),
        "book_asset": asset("book-a.yaml"),
        "joints": joints,
        "book_pose": frames[record["child_frame_id"]],
        "parent_pose": frames[record["parent_frame_id"]],
        "trace": trace,
        "log": str(log),
        "outcomes": str(outcomes),
    }










def build(args):
    import numpy as np
    import torch
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
    from curobo.types import DeviceCfg
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    from ...algorithms import analytic as module
    state = source_state()
    source = Usd.Stage.Open(str(state["book_asset"]))
    collision = mesh_data(source, True)
    visual = mesh_data(source, False)
    bounds = collision.bounds
    request_config = {
        "shape_hint": "solid_box",
        "dimensions_m": collision.extents.tolist(),
        "sphere_budget": 16,
        "task_profile": "narrow_passage",
        "max_outward_offset_m": 0.001,
        "validation_probe_radius_m": 0.006,
    }
    request = module.parse_structure_aware_request(
        request_config, default_budget=16, default_seed=0
    )
    result = module.generate_structure_aware_spheres(
        request, reference_bounds_m=tuple(map(tuple, bounds))
    )
    if result.validation.status.value != "critical_tests_passed":
        raise RuntimeError("Structure-aware validation failed")
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    baseline = fit_spheres_to_mesh(
        collision,
        num_spheres=16,
        fit_type=SphereFitType.MORPHIT,
        compute_metrics=True,
        device_cfg=DeviceCfg(device=args.device),
    )
    if baseline.debug_info.get("containment_enforced"):
        raise RuntimeError("Baseline unexpectedly enabled containment enforcement")
    baseline_spheres = [
        (*c, float(r))
        for c, r in zip(
            baseline.centers.detach().cpu().tolist(),
            baseline.radii.detach().cpu().tolist(),
            strict=True,
        )
    ]
    new_spheres = [(*s.center_m, s.radius_m) for s in result.spheres]
    sets = {"baseline": baseline_spheres, "structure_aware": new_spheres}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    usd_path = output / (args.method + ".usda")
    # All output files belong to this preview; source USD layers are only references.
    stage = (
        Usd.Stage.CreateNew(str(usd_path))
        if not usd_path.exists()
        else Usd.Stage.Open(str(usd_path))
    )
    if stage.GetPrimAtPath("/World"):
        stage.RemovePrim("/World")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1200)
    book_material = material(stage, "/World/Materials/Book", (0.72, 0.74, 0.78), 0.22)
    colors = {"baseline": (1.0, 0.38, 0.05), "structure_aware": (0.1, 0.55, 1.0)}
    methods = list(sets) if args.method == "both" else [args.method]
    metadata = {
        "scope": "static fit comparison, NOT runtime native readback or physics replay",
        "robot_pose_scope": "preceding commanded grasp joints plus close -0.7; not measured snapshot",
        "source_run": str(PROJECT / "run.yaml"),
        "source_log": state["log"],
        "source_joint_targets": state["outcomes"],
        "book_asset": str(state["book_asset"]),
        "book_pose": state["book_pose"],
        "attachment_record": state["trace"]["records"][0],
        "collision_bounds_m": bounds.tolist(),
        "visual_bounds_m": visual.bounds.tolist(),
        "collision_minus_visual_center_m": (
            bounds.mean(0) - visual.bounds.mean(0)
        ).tolist(),
        "structure_config": request_config,
        "baseline_fit_type": str(baseline.fit_type)
        if hasattr(baseline, "fit_type")
        else "MORPHIT",
        "baseline_debug": baseline.debug_info,
        "spheres_object_local_m": sets,
        "structure_validation_status": result.validation.status.value,
    }
    close_targets = []
    for index, method in enumerate(methods):
        panel_path = "/World/" + method
        panel = UsdGeom.Xform.Define(stage, panel_path)
        # Display-only side-by-side offset; underlying world poses remain recorded.
        robot_path = panel_path + "/Robot"
        robot = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
        robot.GetReferences().AddReference(str(state["robot_asset"]))
        worlds = pose_robot(
            stage,
            robot_path,
            state["run"]["scenario"]["scene"]["robots"]["g2"]["pose"],
            state["joints"],
        )
        anchor = next(
            (m for p, m in worlds.items() if p.endswith("/left_gripper_center")), None
        )
        if anchor is not None:
            metadata["robot_reconstruction_anchor_error_m"] = float(
                (
                    anchor.ExtractTranslation()
                    - Gf.Vec3d(*state["parent_pose"]["xyz_m"])
                ).GetLength()
            )
        book_path = panel_path + "/BookAtAttach"
        book = UsdGeom.Xform.Define(stage, book_path).GetPrim()
        book.GetReferences().AddReference(str(state["book_asset"]))
        set_matrix(book, matrix(state["book_pose"]))
        # Hide the original solid collider from rendering; explicit teal wire below remains.
        for p in Usd.PrimRange(book):
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh):
                UsdGeom.Imageable(p).MakeInvisible()
        bind(book, book_material)
        collision_root = UsdGeom.Xform.Define(stage, panel_path + "/CollisionReference")
        set_matrix(collision_root.GetPrim(), matrix(state["book_pose"]))
        wire_mesh(
            stage, str(collision_root.GetPath()) + "/Mesh", collision, (0.08, 0.9, 0.67)
        )
        spheres_root = UsdGeom.Xform.Define(stage, panel_path + "/CollisionSpheres")
        set_matrix(spheres_root.GetPrim(), matrix(state["book_pose"]))
        sphere_material = material(
            stage, "/World/Materials/" + method, colors[method], 0.7
        )
        for i, (x, y, z, r) in enumerate(sets[method]):
            sphere = UsdGeom.Sphere.Define(
                stage, str(spheres_root.GetPath()) + f"/sphere_{i:02d}"
            )
            sphere.CreateRadiusAttr(r)
            sphere.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            bind(sphere.GetPrim(), sphere_material)
        # Initial location is a second, thin outline; no fake duplicate physical book.
        original = UsdGeom.Xform.Define(stage, panel_path + "/BookInitialPosition")
        set_matrix(
            original.GetPrim(),
            matrix(state["run"]["scenario"]["scene"]["objects"]["book_a"]["pose"]),
        )
        wire_mesh(
            stage,
            str(original.GetPath()) + "/VisualOutline",
            visual,
            (0.8, 0.25, 0.85),
            0.00025,
        )
        panel.AddTranslateOp().Set(Gf.Vec3d(index * 1.7, 0, 0))
        center = np.asarray(state["book_pose"]["xyz_m"]) + np.array([index * 1.7, 0, 0])
        close_targets.append(center)
    # Static-only preview: even clicking Play cannot make these display copies move.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    target = np.mean(close_targets, axis=0)
    distance = 2.6 if len(methods) == 2 else 0.7
    eye = target + np.array([distance * 0.6, -distance, distance * 0.6])
    camera = UsdGeom.Camera.Define(stage, "/World/CameraClose")
    set_matrix(
        camera.GetPrim(),
        Gf.Matrix4d()
        .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        .GetInverse(),
    )
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100))
    camera.CreateFocalLengthAttr(24)
    wide = UsdGeom.Camera.Define(stage, "/World/CameraRobot")
    robot_eye = target + np.array([2, -3, 1.6])
    set_matrix(
        wide.GetPrim(),
        Gf.Matrix4d()
        .SetLookAt(Gf.Vec3d(*robot_eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        .GetInverse(),
    )
    wide.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100))
    stage.GetRootLayer().Save()
    metadata["usd"] = str(usd_path)
    metadata["counts"] = {k: len(v) for k, v in sets.items()}
    (output / (args.method + ".json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in (
                    "usd",
                    "counts",
                    "collision_minus_visual_center_m",
                    "robot_reconstruction_anchor_error_m",
                )
                if k in metadata
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return usd_path




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("baseline", "structure_aware", "both"), default="both"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--view", choices=("close", "robot"), default="close")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Generate static USD and evidence JSON without launching Isaac",
    )
    parser.add_argument(
        "--open-existing",
        action="store_true",
        help="Open previously generated USD without rerunning fitters",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Bounded renderer smoke test; use with --frames",
    )
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.headless and args.frames < 1:
        parser.error("--headless requires a positive --frames bound")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # Kit extensions need the Conda C++ runtime, not Ubuntu's older libstdc++.
    library_dir = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != library_dir:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = library_dir + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        os.execvpe(sys.executable, [sys.executable, "-m", "asset_collision_spheres.adapters.workspace.book", *sys.argv[1:]], env)
    path = (
        args.output.resolve() / (args.method + ".usda")
        if args.open_existing
        else build(args)
    )
    if not path.is_file():
        parser.error(f"Missing preview: {path}")
    if not args.build_only:
        open_isaac(path, args)


if __name__ == "__main__":
    main()
