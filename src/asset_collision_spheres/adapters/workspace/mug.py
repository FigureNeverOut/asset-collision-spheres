"""1e handled-mug static diagnostic; not physics replay or planner readback.

Baseline/budget modes use public cuRobo MORPHIT, including native fallbacks.
Structure-aware mode uses the experimental full-mesh region generator; it does
not use the analytic, handleless-cup specialization or change the runtime task.
Robot posture is reconstructed by joint-limited static IK at the authored lift
target. It is not a measured trajectory, nor a collision-checked IK solution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from ...paths import workspace_root, output_root, config_path

from .book import (
    bind,
    material,
    matrix,
    mesh_data,
    open_isaac,
    pose_robot,
    set_matrix,
    wire_mesh,
)

ROOT = workspace_root()
RUN = ROOT / "FastSim-Demo/task/scenario_type_1/1e/run.yaml"
CATALOG = ROOT / "FastSim-Demo/.fastsim-demo/runtime/catalog"
LEGACY_OUTPUT = ROOT / "task2sim/runs/1e/mug_sphere_preview"
DEFAULT_OUTPUT = output_root() / "mug/sphere_preview"
DEFAULT_STRUCTURE_CONFIG = (
    config_path("mug/1e_mug_mesh_regions.json")
)


def inline_pose(data):
    w, x, y, z = data["quaternion"]
    return {"xyz_m": data["position"], "quat_xyzw": [x, y, z, w]}


def find_solver_config(value):
    if isinstance(value, dict):
        if "attached_object_sphere_budget" in value:
            return value
        for child in value.values():
            result = find_solver_config(child)
            if result is not None:
                return result
    return None


def load_state():
    import hashlib

    import yaml

    run = yaml.safe_load(RUN.read_text())
    scene = run["scenario"]["scene"]
    tasks = {t["name"]: t for t in run["scenario"]["behavior"]["task"]}
    assets, components = {}, {}
    for alias, filename in (("robot", "g2-crsb-swiftpicker"), ("mug", "mug-000")):
        component = yaml.safe_load((CATALOG / (filename + ".yaml")).read_text())
        resource = component["variants"]["isaaclab"]["resources"]["model"]
        path = Path(resource["uri"])
        if not path.is_file():
            raise RuntimeError(f"Resolved 1e runtime asset is unavailable: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != resource["sha256"]:
            raise RuntimeError(f"Runtime catalog hash mismatch: {path}")
        assets[alias] = path
        components[alias] = component
    for alias, entity in (
        ("mug", scene["objects"]["mug"]),
        ("robot", scene["robots"]["g2"]),
    ):
        if components[alias]["id"] != entity["use"]:
            raise RuntimeError(f"1e catalog does not match the configured {alias}")
        if entity.get("scale", [1, 1, 1]) != [1, 1, 1]:
            raise RuntimeError("This static viewer requires unit entity scale")
    joints = {
        k: float(v["position"])
        for k, v in components["robot"]["defaults"]["initial_state"]["joints"].items()
    }
    joints.update(
        {
            k: float(v["position"])
            for k, v in scene["robots"]["g2"]
            .get("initial_state", {})
            .get("joints", {})
            .items()
        }
    )
    joints.update(tasks["close-right-gripper"]["motions"]["close"]["joints"])
    grasp = tasks["move-to-mug-grasp"]["motions"]["move"]["target"]["pose"]
    lift = tasks["lift-mug-10cm"]["motions"]["move"]["target"]["pose"]
    if (
        grasp["frame"] != "object"
        or grasp["object"] != "mug"
        or lift["frame"] != "world"
    ):
        raise RuntimeError("1e grasp/lift frame semantics have changed")
    return {
        "assets": assets,
        "joints": joints,
        "robot_pose": scene["robots"]["g2"]["pose"],
        "initial_pose": scene["objects"]["mug"]["pose"],
        "grasp_pose": inline_pose(grasp["data"][0]),
        "lift_pose": inline_pose(lift["data"][0]),
        "solver_config": find_solver_config(run),
    }


def static_ik(stage, robot_path, root_pose, positions, target_pose):
    """Solve only the seven right arm joints using authored USD joint frames."""
    import numpy as np
    from pxr import Gf, Usd, UsdPhysics
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    def frame(pos, quat):
        m = Gf.Matrix4d(1)
        m.SetRotate(Gf.Quatd(quat))
        m.SetTranslateOnly(Gf.Vec3d(pos))
        return m

    links = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            children = joint.GetBody1Rel().GetTargets()
            if children:
                key = str(children[0])
                if key in links:
                    raise RuntimeError(
                        "Static IK requires a tree, not closed-chain joints"
                    )
                links[key] = joint
    end = next(k for k in links if k.endswith("/right_gripper_center"))
    chain, current = [], end
    while current is not None:
        joint = links[current]
        prim = joint.GetPrim()
        f0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        f1 = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        axis = prim.GetAttribute("physics:axis").Get()
        vector = {
            "X": Gf.Vec3d(1, 0, 0),
            "Y": Gf.Vec3d(0, 1, 0),
            "Z": Gf.Vec3d(0, 0, 1),
        }.get(axis)
        chain.append((prim, f0, f1.GetInverse(), vector))
        parents = joint.GetBody0Rel().GetTargets()
        current = str(parents[0]) if parents else None
    chain.reverse()
    names = [f"idx{61 + i}_arm_r_joint{i + 1}" for i in range(7)]
    by_name = {p.GetName(): p for p, *_ in chain}
    lower = np.radians(
        [float(by_name[n].GetAttribute("physics:lowerLimit").Get()) for n in names]
    )
    upper = np.radians(
        [float(by_name[n].GetAttribute("physics:upperLimit").Get()) for n in names]
    )
    target = np.array(matrix(target_pose)).T

    def forward(q):
        values = {**positions, **dict(zip(names, q, strict=True))}
        world = matrix(root_pose)
        for prim, f0, f1_inv, vector in chain:
            motion = Gf.Matrix4d(1)
            value = values.get(prim.GetName(), 0)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                motion.SetRotate(Gf.Rotation(vector, math.degrees(value)))
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                motion.SetTranslate(vector * value)
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise RuntimeError(f"Unsupported static IK joint: {prim.GetPath()}")
            world = f1_inv * motion * f0 * world
        return np.array(world).T

    def residual(q):
        actual = forward(q)
        rotation = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
        return np.r_[actual[:3, 3] - target[:3, 3], rotation * 0.2]

    start = np.clip([positions[n] for n in names], lower + 1e-8, upper - 1e-8)
    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=500,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    error = residual(result.x)
    translation_error = float(np.linalg.norm(error[:3]))
    rotation_error = float(np.linalg.norm(error[3:]) / 0.2)
    # Fixed-base preview: keep and report the sub-mm / 0.3 degree residual;
    # unlike the mission solver, this IK does not move the mobile base.
    if translation_error > 0.001 or rotation_error > 0.005:
        raise RuntimeError(
            f"Static IK did not reach the target: {translation_error} m, {rotation_error} rad"
        )
    return {**positions, **dict(zip(names, result.x.tolist(), strict=True))}, {
        "translation_error_m": translation_error,
        "rotation_error_rad": rotation_error,
        "scope": "joint-limited static IK; no environment/self collision validation or physics",
    }


def assert_aligned(collision, visual, tolerance=2e-6):
    from scipy.spatial import cKDTree

    # Bounds alone can miss displacements on a symmetric surface. Verify both
    # vertex sets as well: this 1e asset has matching visual/collision topology.
    error = max(
        float(cKDTree(a.vertices).query(b.vertices)[0].max())
        for a, b in ((collision, visual), (visual, collision))
    )
    if error > tolerance:
        raise RuntimeError(
            f"Visual/collision vertices do not align ({error:.6g} m); refusing a misleading preview"
        )
    return {
        "visual_collision_vertex_error_m": error,
        "collision_minus_visual_center_m": (
            collision.bounds.mean(0) - visual.bounds.mean(0)
        ).tolist(),
        "bounds_m": collision.bounds.tolist(),
        "extents_m": collision.extents.tolist(),
    }


def fit_baseline(mesh, count, device, seed):
    import numpy as np
    import torch
    import trimesh
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
    from curobo.types import DeviceCfg

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result = fit_spheres_to_mesh(
        mesh,
        num_spheres=count,
        fit_type=SphereFitType.MORPHIT,
        compute_metrics=True,
        device_cfg=DeviceCfg(device=device),
    )
    if result.debug_info.get("containment_enforced"):
        raise RuntimeError("Containment debug behavior must not be enabled")
    centers = result.centers.detach().cpu().numpy()
    radii = result.radii.detach().cpu().numpy().reshape(-1)
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(radii).all()
        or np.any(radii <= 0)
    ):
        raise RuntimeError("Invalid fitted spheres")
    samples, _ = trimesh.sample.sample_surface(mesh, 20000, seed=seed)
    gap = (
        np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
        - radii[None, :]
    ).min(1)
    return np.c_[centers, radii].tolist(), {
        "requested_count": count,
        "actual_count": len(radii),
        "radius_range_m": [float(radii.min()), float(radii.max())],
        "surface_probe_count": len(samples),
        "sampled_surface_coverage_fraction": float(np.mean(gap <= 0)),
        "sampled_max_surface_gap_m": float(max(0, gap.max())),
        "diagnostic_only": "random surface samples, NOT task collision acceptance or cavity-clearance certification",
        "debug_info": result.debug_info,
    }




def load_mesh_region_generator():
    from ...algorithms import mesh_regions
    return mesh_regions


def fit_structure_aware(mesh, state, args):
    import hashlib

    config_path = args.structure_config.resolve()
    config = json.loads(config_path.read_text())
    actual_sha = hashlib.sha256(state["assets"]["mug"].read_bytes()).hexdigest()
    if config.get("asset_sha256") != actual_sha:
        raise RuntimeError("Mesh-region configuration belongs to a different mug asset")
    if config["sphere_budget"] != args.diagnostic_budget:
        raise RuntimeError(
            "For structure-aware mode, region quotas and --diagnostic-budget must match"
        )
    module = load_mesh_region_generator()
    result = module.generate_mesh_region_spheres(mesh, config)
    info = dict(result.diagnostics)
    info["structure_config_path"] = str(config_path)
    info["structure_config"] = config
    info["holdout"] = module.evaluate_mesh_spheres(mesh, result.spheres, config)
    # Re-evaluate saved baseline fits with EXACTLY the same independent probes.
    # Do not refit or overwrite their screenshots while making the new preview.
    comparisons = {}
    for method in ("baseline", "budget"):
        saved = args.output.resolve() / (method + ".json")
        if not saved.is_file():
            continue
        previous = json.loads(saved.read_text())
        if previous["assets"]["mug"] != str(state["assets"]["mug"]):
            raise RuntimeError("Saved comparison belongs to a different mug asset")
        comparisons[method] = module.evaluate_mesh_spheres(
            mesh, previous["spheres_object_local_m"], config
        )
        comparisons[method]["source_json"] = str(saved)
        comparisons[method]["actual_count"] = len(previous["spheres_object_local_m"])
    info["saved_baseline_holdouts"] = comparisons
    info["acceptance_warning"] = (
        "Static experiment only. Keep failed hold-out visible; do NOT deploy to native attachment without validation."
    )
    return result.spheres.tolist(), info


def build(args, fit_override=None):
    import hashlib

    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    state = load_state()
    mug_stage = Usd.Stage.Open(str(state["assets"]["mug"]))
    if (
        not math.isclose(UsdGeom.GetStageMetersPerUnit(mug_stage), 1.0)
        or UsdGeom.GetStageUpAxis(mug_stage) != "Z"
    ):
        raise RuntimeError("1e runtime mug must already be normalized to Z-up metres")
    collision, visual = mesh_data(mug_stage, True), mesh_data(mug_stage, False)
    alignment = assert_aligned(collision, visual)
    budget = int(state["solver_config"]["attached_object_sphere_budget"])
    count = budget if args.method == "baseline" else args.diagnostic_budget
    if fit_override is not None:
        spheres, fit_info = fit_override(collision, state, args)
    elif args.method == "reference":
        spheres, fit_info = [], {"actual_count": 0}
    elif args.method == "structure_aware":
        spheres, fit_info = fit_structure_aware(collision, state, args)
    else:
        spheres, fit_info = fit_baseline(
            collision, count, args.device, int(state["solver_config"].get("seed", 5))
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.method}.usda"
    stage = (
        Usd.Stage.CreateNew(str(path))
        if not path.exists()
        else Usd.Stage.Open(str(path))
    )
    if stage.GetPrimAtPath("/World"):
        stage.RemovePrim("/World")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1100)
    robot_path = "/World/Robot"
    robot = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
    robot.GetReferences().AddReference(str(state["assets"]["robot"]))
    joints, ik_info = static_ik(
        stage, robot_path, state["robot_pose"], state["joints"], state["lift_pose"]
    )
    worlds = pose_robot(stage, robot_path, state["robot_pose"], joints)
    anchor = next(m for p, m in worlds.items() if p.endswith("/right_gripper_center"))
    # Gf uses row-vector matrices: T_world_mug = inv(T_mug_grasp) * T_world_grasp.
    mug_world = matrix(state["grasp_pose"]).GetInverse() * anchor
    anchor_error = float(
        np.abs(
            np.array(matrix(state["grasp_pose"]) * mug_world) - np.array(anchor)
        ).max()
    )
    if anchor_error > 1e-8:
        raise RuntimeError("Attachment frame composition failed")
    mug = UsdGeom.Xform.Define(stage, "/World/MugAtAttach").GetPrim()
    mug.GetReferences().AddReference(str(state["assets"]["mug"]))
    set_matrix(mug, mug_world)
    for prim in Usd.PrimRange(mug):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(prim).MakeInvisible()
    if args.method != "reference":
        bind(mug, material(stage, "/World/Materials/Mug", (0.7, 0.8, 0.85), 0.18))
    if args.hide_visual:
        UsdGeom.Imageable(mug).MakeInvisible()
    ref = UsdGeom.Xform.Define(stage, "/World/CollisionReference")
    set_matrix(ref.GetPrim(), mug_world)
    wire_mesh(
        stage, "/World/CollisionReference/Mesh", collision, (0.08, 0.85, 0.7), 0.000055
    )
    # Optional raw mesh inspection via the Stage eye; hidden to avoid 25k-edge clutter.
    UsdGeom.Imageable(ref).MakeInvisible()
    sphere_root = UsdGeom.Xform.Define(stage, "/World/CollisionSpheres")
    set_matrix(sphere_root.GetPrim(), mug_world)
    color = (1, 0.42, 0.05) if args.method == "baseline" else (0.08, 0.52, 1)
    sphere_mat = material(stage, "/World/Materials/Spheres", color, 0.8)
    region_colors = {
        "wall": (0.1, 0.8, 0.35),
        "rim": (1.0, 0.7, 0.05),
        "bottom": (0.12, 0.45, 1.0),
        "handle": (1.0, 0.3, 0.08),
    }
    if fit_override is not None:
        import colorsys

        labels = sorted(set(fit_info.get("sphere_regions", [])))
        region_colors = {
            name: region_colors.get(
                name, colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.7, 0.95)
            )
            for i, name in enumerate(labels)
        }
        if fit_info.get("mode") in {"convex_surface", "original_surface"}:
            from ...preview.convex import COLORS
            region_colors = {name: COLORS[name] for name in labels}
    region_materials = (
        {
            name: material(stage, "/World/Materials/Region_" + name, rgb, 0.85)
            for name, rgb in region_colors.items()
        }
        if args.method == "structure_aware" or fit_override is not None
        else {}
    )
    sphere_regions = fit_info.get("sphere_regions", ["generic"] * len(spheres))
    for i, (x, y, z, r) in enumerate(spheres):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/CollisionSpheres/sphere_{i:03d}")
        sphere.CreateRadiusAttr(r)
        sphere.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        region = sphere_regions[i]
        sphere.GetPrim().SetCustomDataByKey("structureRegion", region)
        bind(sphere.GetPrim(), region_materials.get(region, sphere_mat))
    original = UsdGeom.Xform.Define(stage, "/World/MugInitialPosition")
    set_matrix(original.GetPrim(), matrix(state["initial_pose"]))
    wire_mesh(
        stage, "/World/MugInitialPosition/Outline", visual, (0.8, 0.2, 0.85), 0.00006
    )
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    # Both the cup's initial and attached locations are visible in the close view.
    center_local = Gf.Vec3d(*map(float, collision.bounds.mean(0)))
    target = np.array(mug_world.Transform(center_local))
    camera_offsets = {
        "Close": [0.32, -0.4, 0.26],
        "Robot": [1.4, -2.2, 0.9],
        "Top": [0.04, -0.04, 0.5],
        "Side": [0.45, -0.1, 0.04],
    }
    for view, offset in camera_offsets.items():
        camera = UsdGeom.Camera.Define(stage, "/World/Camera" + view)
        look = target.copy()
        if view == "Close":
            look[2] -= 0.025
        if view == "Robot":
            look = np.array(state["robot_pose"]["xyz_m"]) + [0, -0.35, 0.85]
        eye = look + offset
        set_matrix(
            camera.GetPrim(),
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1))
            .GetInverse(),
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100))
        camera.CreateFocalLengthAttr(32 if view != "Robot" else 24)
    stage.GetRootLayer().Save()
    evidence = {
        "scope": __doc__,
        "method": fit_info.get("mode") if fit_override is not None else (
            "experimental full-mesh structure-aware region fit"
            if args.method == "structure_aware"
            else "reference only"
            if args.method == "reference"
            else "public MORPHIT request, including its native fallbacks; see fit.debug_info"
        ),
        "analytic_cup_specialization_used": False,
        "mesh_region_specialization_used": args.method == "structure_aware",
        "unsupported_specialization_reason": "1e mug has a handle; the analytic cup implementation is handleless-only",
        "runtime_budget_unchanged": budget,
        "run_yaml": str(RUN),
        "run_sha256": hashlib.sha256(RUN.read_bytes()).hexdigest(),
        "assets": {k: str(v) for k, v in state["assets"].items()},
        "alignment": alignment,
        "ik": ik_info,
        "joints_rad_or_m": joints,
        "attachment_composition_error": anchor_error,
        "world_from_mug_column_matrix": np.array(mug_world).T.tolist(),
        "initial_pose": state["initial_pose"],
        "spheres_object_local_m": spheres,
        "fit": fit_info,
        "usd": str(path),
        "physics_and_planning_executed": False,
    }
    (output / (args.method + ".json")).write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    printed = {k: evidence[k] for k in ("usd", "alignment", "ik", "fit")}
    if fit_info.get("mode") in {"convex_surface", "original_surface"}:
        printed["fit"] = {k: fit_info[k] for k in (
            "algorithm_version", "actual_count", "sampling_strategy", "radius_m", "count_stop_reason", "review_required",
        )}
        printed["full_report"] = str(output / (args.method + ".json"))
    if fit_info.get("algorithm_version") == "auto-geometry-joint-v2-experimental":
        printed["fit"] = {k: fit_info[k] for k in (
            "algorithm_version", "actual_sphere_count", "max_outward_offset_m", "radius_range_m",
            "count_stop_reason", "review_required",
        )}
        printed["full_report"] = str(output / (args.method + ".json"))
    print(
        json.dumps(
            printed,
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("baseline", "budget", "reference", "structure_aware"),
        default="baseline",
    )
    parser.add_argument(
        "--diagnostic-budget",
        type=int,
        default=64,
        help="Diagnostic only; does not change 1e runtime capacity",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--structure-config", type=Path, default=DEFAULT_STRUCTURE_CONFIG
    )
    parser.add_argument(
        "--view", choices=("close", "robot", "top", "side"), default="close"
    )
    parser.add_argument(
        "--hide-visual",
        action="store_true",
        help="Hide attached cup; with --open-existing this is a session-only display change",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--open-existing", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    if not 1 <= args.diagnostic_budget <= 256:
        parser.error("Diagnostic budget must be in [1, 256]")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    library_dir = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != library_dir:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = library_dir + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        os.execvpe(sys.executable, [sys.executable, "-m", "asset_collision_spheres.adapters.workspace.mug", *sys.argv[1:]], env)
    path = (
        args.output.resolve() / (args.method + ".usda")
        if args.open_existing
        else build(args)
    )
    if not path.is_file():
        parser.error(f"Missing preview: {path}; first run without --open-existing")
    if not args.build_only:
        # Reuse the same renderer, but supply task-specific camera and legend.
        args.camera_path = "/World/Camera" + args.view.capitalize()
        args.hidden_paths = ["/World/MugAtAttach"] if args.hide_visual else []
        args.preview_legend = "1e STATIC | orange=current budget, blue=larger budget SAME fitter; actual counts in JSON; translucent=attached mug, purple=initial mug; NOT planner acceptance"
        if args.method == "structure_aware":
            args.preview_legend = "1e EXPERIMENTAL STRUCTURE | green=wall, yellow=rim, blue=bottom, orange=handle, purple=initial mug; actual count in JSON; NOT planner acceptance"
        open_isaac(path, args)


if __name__ == "__main__":
    main()
