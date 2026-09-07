"""Separate 1e comparison: frozen manual / global greedy / geometry-balanced.

Builds a static posture, then checks real cuRobo attachment and renders its FK
readback. Not a physics replay or a successful full-task trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from ...paths import workspace_root, output_root, config_path

from . import mug as preview

ROOT = preview.ROOT
LEGACY_OUTPUT = ROOT / "task2sim/runs/1e/mug_auto_geometry"
OUTPUT = output_root() / "mug/mug_auto_geometry"
FROZEN = preview.LEGACY_OUTPUT / "structure_aware.json"
FROZEN_SHA = "789ef30406fbdd231375166d7fa01c50e1b4dac3adc25d0bbb80ebcd91e093df"
BACKEND = "fastsim_plugin_mission.task.unisolver.backends.curobo"


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def write_runtime_config(args):
    """Generate a separate 1e run; change only sphere settings in CuRobo profiles."""
    import yaml

    run = yaml.safe_load(preview.RUN.read_text())
    if args.method == "manual_frozen":
        artifact = args.output / (args.method + "_native_input.json")
        config = {
            "mode": "precomputed",
            "artifact_path": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    else:
        config = json.loads(args.geometry_config.read_text())
        config["mode"] = args.method
        # FastSim's portable schema subset has no nullable numeric type. Omit
        # optional nulls in authored YAML; the fitter treats absent == None.
        config = {key: value for key, value in config.items() if value is not None}
    changed = []

    def visit(value, parts=()):
        if isinstance(value, dict):
            if value.get("backend") == "curobo" and isinstance(
                value.get("config"), dict
            ):
                value["config"].update(
                    attached_object_sphere_budget=args.diagnostic_budget,
                    attached_object_sphere_method="mesh",
                    attached_object_sphere_fallback="error",
                    attached_object_mesh_configs={"mug": config},
                )
                changed.append("/".join(parts))
            for key, child in value.items():
                visit(child, (*parts, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*parts, str(index)))

    visit(run)
    if not changed:
        raise RuntimeError(
            "No CuRobo profile found; do not produce an ineffective experiment config"
        )
    target = args.output / ("run." + args.method + ".yaml")
    target.write_text(
        "# Generated isolated sphere experiment; original run unchanged.\n"
        + yaml.safe_dump(run, sort_keys=False, allow_unicode=True)
    )
    dump(
        args.output / ("run." + args.method + ".provenance.json"),
        {
            "source": str(preview.RUN),
            "source_sha256": hashlib.sha256(preview.RUN.read_bytes()).hexdigest(),
            "modified_profiles": changed,
            "modified_fields": [
                "attached_object_sphere_budget",
                "attached_object_sphere_method",
                "attached_object_sphere_fallback",
                "attached_object_mesh_configs",
            ],
            "full_physics_task_executed": False,
        },
    )
    return target


def materialize_offline_run(path):
    """Use the existing Demo URI resolver only: no SVN update or asset cooking."""
    import importlib.util

    source = ROOT / "FastSim-Demo/tools/svn_assets.py"
    spec = importlib.util.spec_from_file_location(
        "fastsim_mug_offline_resolver", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.materialize_run_config(
        path,
        module.DEFAULT_ASSET_WORKING_COPY,
        runtime_root=path.parent / "offline_runtime",
    )


def fit(mesh, state, args):
    from asset_collision_spheres.adapters.fastsim import (
        check_spheres,
        mesh_fingerprint,
    )

    policy_config = json.loads(args.geometry_config.read_text()) if args.method != "manual_frozen" else {}
    if "geometry_policy" in policy_config:
        from ...adapters.fastsim import prepare_request, fit_mesh_attachment
        config, _ = prepare_request({**policy_config, "mode":args.method}, capacity=args.diagnostic_budget, seed=5)
        result = fit_mesh_attachment(mesh,config)
        info = {**result.diagnostics, "sphere_regions":list(result.regions)}
        args.output.mkdir(parents=True,exist_ok=True)
        args.native_artifact = args.output / (args.method + "_native_input.json")
        dump(args.native_artifact, {"schema":"asset-collision-spheres/geometry-native-input/1",
                                   "mesh_sha256":mesh_fingerprint(mesh),"config":config,
                                   "spheres_object_local_m":result.spheres.tolist()})
        return result.spheres.tolist(), info
    if args.method == "convex_surface":
        from ...algorithms.convex_surface import generate_convex_surface_spheres
        from ...adapters.fastsim import prepare_request
        config = {**json.loads(args.geometry_config.read_text()), "mode": "convex_surface"}
        config, _ = prepare_request(config, capacity=args.diagnostic_budget, seed=5)
        result = generate_convex_surface_spheres(mesh, config)
        info = {**result.diagnostics, "sphere_regions": list(result.regions)}
        args.output.mkdir(parents=True, exist_ok=True)
        args.native_artifact = args.output / "convex_surface_native_input.json"
        # Provenance only; actual backend invokes the generator, not precomputed.
        dump(args.native_artifact, {"schema": "asset-collision-spheres/convex-native-input/1",
                                   "mesh_sha256": mesh_fingerprint(mesh), "config": config,
                                   "spheres_object_local_m": result.spheres.tolist()})
        return result.spheres.tolist(), info

    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError(
            "Approved manual artifact changed; refuse to silently replace the comparison"
        )
    frozen = json.loads(FROZEN.read_text())
    if frozen["assets"]["mug"] != str(state["assets"]["mug"]):
        raise RuntimeError("Frozen comparison belongs to a different asset")
    if args.method == "manual_frozen":
        spheres = frozen["spheres_object_local_m"]
        info = {
            "mode": "manual_frozen",
            "sphere_regions": frozen["fit"]["sphere_regions"],
            "source_sha256": FROZEN_SHA,
            "generation_time_s": 0.0,
            "manual_region_caps_and_quotas_used": True,
            "candidate_pool_differs_from_B_C": True,
        }
        expansion = frozen["fit"]["max_outward_offset_m"]
    else:
        from asset_collision_spheres.algorithms.auto_geometry import (
            generate_auto_geometry_spheres,
        )

        config = json.loads(args.geometry_config.read_text())
        config["mode"] = args.method
        from ...algorithms.joint_selection import is_extended
        if is_extended(config):
            from ...adapters.fastsim import prepare_request
            config, _ = prepare_request(config, capacity=args.diagnostic_budget, seed=5)
        elif config["sphere_budget"] != args.diagnostic_budget:
            raise RuntimeError(
                "Explicit geometry config budget must match --diagnostic-budget"
            )
        result = generate_auto_geometry_spheres(mesh, config)
        spheres, info = result.spheres.tolist(), dict(result.diagnostics)
        info["sphere_regions"] = list(result.regions)
        expansion = info["max_outward_offset_m"]
    check_spheres(mesh, spheres, args.diagnostic_budget, expansion)
    info["actual_count"] = len(spheres)
    info["max_outward_offset_m"] = expansion
    # Only the evaluator sees manual labels and paths; none enter B/C generation.
    evaluator = preview.load_mesh_region_generator()
    reference_config = json.loads(preview.DEFAULT_STRUCTURE_CONFIG.read_text())
    info["heldout_manual_labels_only"] = evaluator.evaluate_mesh_spheres(
        mesh, spheres, reference_config
    )
    info["ordinary_misses_block_attachment"] = False
    artifact = {
        "schema": "fastsim/mesh-spheres/1",
        "mesh_sha256": mesh_fingerprint(mesh),
        "max_outward_offset_m": expansion,
        "spheres_object_local_m": spheres,
        "sphere_regions": info["sphere_regions"],
        "source_mode": args.method,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    args.native_artifact = args.output / (args.method + "_native_input.json")
    dump(args.native_artifact, artifact)
    return spheres, info


def native_check(args, path):
    """Use real Mission backend + native planner collision checker, in isolated scene."""
    import numpy as np
    import torch
    import yaml
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo.scene import Cuboid as NativeCuboid
    from curobo.scene import Scene
    from fastsim_plugin_mission.task.unisolver.arm import ArmPlanningContext
    from fastsim_plugin_mission.task.unisolver.backends.curobo.backend import (
        CuroboBackend,
    )
    from fastsim_plugin_mission.task.unisolver.backends.curobo.config import (
        CuroboConfig,
    )
    from fastsim_plugin_mission.task.unisolver.core import (
        AttachedObject,
        CollisionElement,
        CollisionLink,
        CollisionObject,
        CollisionPolicy,
        CollisionWorld,
        FrameId,
        JointState,
        Ok,
        Pose,
        RobotAttachmentState,
        RobotRootState,
        Transform,
        TriangleMesh,
    )
    from pxr import Gf, Usd, UsdGeom
    from scipy.spatial.transform import Rotation

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Native verification requires CUDA; --skip-native is explicitly visual-only"
        )
    evidence_path = path.with_suffix(".json")
    evidence = json.loads(evidence_path.read_text())
    state = preview.load_state()
    mesh = preview.mesh_data(Usd.Stage.Open(str(state["assets"]["mug"])), True)
    encoded = mesh.export(file_type="obj", digits=12).encode()
    mesh_path = args.output / (hashlib.sha256(encoded).hexdigest() + ".obj")
    mesh_path.write_bytes(encoded)
    robot_config = (
        ROOT
        / "FastSim-Demo/.fastsim-demo/svn/benchmark/trunk/data/robots/G2/configs/curobo/mobile_manipulation_right.yml"
    )
    raw_robot = yaml.safe_load(robot_config.read_text())["robot_cfg"]["kinematics"]
    joints = evidence["joints_rad_or_m"]
    locked = {
        name: joints.get(name, value)
        for name, value in raw_robot["lock_joints"].items()
    }
    frame = FrameId.world()
    rp = state["robot_pose"]
    root = RobotRootState(
        "g2",
        Pose(frame, tuple(rp["xyz_m"]), tuple(np.roll(rp["quat_xyzw"], 1))),
        revision=1,
    )
    mw = np.array(evidence["world_from_mug_column_matrix"])
    mug_pose = Pose(
        frame,
        tuple(mw[:3, 3]),
        tuple(np.roll(Rotation.from_matrix(mw[:3, :3]).as_quat(), 1)),
    )
    gp = state["grasp_pose"]
    relative = Transform(
        tuple(gp["xyz_m"]), tuple(np.roll(gp["quat_xyzw"], 1))
    ).inverse()
    payload = CollisionObject(
        "mug",
        mug_pose,
        (
            CollisionLink(
                "mug-root",
                (CollisionElement("mug/collision", TriangleMesh(mesh_path, bounding_radius=float(np.linalg.norm(mesh.vertices, axis=1).max()))),),
            ),
        ),
        1,
    )
    attachment = AttachedObject("mug", "g2", "right_gripper_center", relative)
    context = ArmPlanningContext(
        1,
        CollisionWorld(frame, 1, (payload,)),
        root,
        RobotAttachmentState("g2", 1, (attachment,)),
    )
    # First manual path uses approved precomputed data. B/C exercise the generator
    # inside the actual backend (not just their exported precomputed copies).
    if args.method == "manual_frozen":
        mesh_config = {
            "mode": "precomputed",
            "artifact_path": str(args.native_artifact),
            "artifact_sha256": hashlib.sha256(
                args.native_artifact.read_bytes()
            ).hexdigest(),
        }
    else:
        mesh_config = json.loads(args.geometry_config.read_text())
        mesh_config["mode"] = args.method
    backend = CuroboBackend(
        CuroboConfig(
            robot_config=robot_config,
            locked_joints=locked,
            device="cuda",
            seed=5,
            attached_object_sphere_budget=args.diagnostic_budget,
            attached_object_sphere_method="mesh",
            attached_object_sphere_fallback="error",
            attached_object_mesh_configs={"mug": mesh_config},
            use_cuda_graph=False,
            warmup=False,
            ik_num_seeds=4,
        )
    )
    try:
        assert backend.apply_arm_planning_context(context) == Ok(None)
        joint_state = JointState(
            "g2",
            frame,
            backend.plannable_joint_names,
            tuple(joints[name] for name in backend.plannable_joint_names),
        )
        bindings, solver = backend._ensure_motion(
            context.collision_world, root, CollisionPolicy()
        )
        native_state = backend._joint_state(
            joint_state, bindings, expected_names=tuple(solver.kinematics.joint_names)
        )
        torch.cuda.synchronize()
        sync_started = time.perf_counter()
        attestation = backend._sync_motion_attachment(
            bindings, solver, joint_state, native_state, root
        )
        torch.cuda.synchronize()
        first_sync_time = time.perf_counter() - sync_started
        assert attestation is not None
        report = asdict(attestation)
        report["requested_mesh_config"] = mesh_config
        report["first_sync_including_fit_diagnostics_fk_time_s"] = first_sync_time
        report["input_artifact_sha256"] = hashlib.sha256(
            args.native_artifact.read_bytes()
        ).hexdigest()
        report["mesh_diagnostics"] = backend._attached_sphere_fit.mesh_diagnostics
        native_world = np.array(
            [(*s.center_world_m, s.radius_m) for s in attestation.world_spheres]
        )
        original = np.array(evidence["spheres_object_local_m"])
        expected_world = original[:, :3] @ mw[:3, :3].T + mw[:3, 3]
        closure = float(
            np.max(np.linalg.norm(native_world[:, :3] - expected_world, axis=1))
        )
        radius_error = float(np.max(np.abs(native_world[:, 3] - original[:, 3])))
        report["preview_vs_native_center_max_m"] = closure
        report["preview_vs_native_radius_max_m"] = radius_error
        if closure > 2e-5 or not np.allclose(
            native_world[:, 3], original[:, 3], atol=1e-7, rtol=0
        ):
            raise RuntimeError(
                f"Native FK differs from static object geometry: centres={closure} m, radii={radius_error} m"
            )
        params = solver.attachment_manager.kinematics_params
        link_index = params.link_name_to_idx_map["attached_object"]
        indices = (params.link_sphere_idx_map == link_index).cpu().numpy().reshape(-1)
        kin = solver.kinematics.compute_kinematics(native_state)
        all_spheres = kin.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        body = all_spheres[(~indices) & (all_spheres[:, 3] > 0)]
        carried = all_spheres[indices & (all_spheres[:, 3] > 0)]
        clearances = (
            np.linalg.norm(carried[:, None, :3] - body[None, :, :3], axis=2)
            - body[None, :, 3]
        ).min(1)
        sentinel = carried[int(np.argmax(clearances)), :3]
        halfsize = 0.001
        if clearances.max() <= np.sqrt(3) * halfsize:
            raise RuntimeError(
                "No payload-only sentinel point clear of robot body spheres"
            )
        # Deliberately isolated test world: no physics or existing task obstacles changed.
        solver.update_world(
            Scene(
                cuboid=[
                    NativeCuboid(
                        name="payload_only_sentinel",
                        dims=[2 * halfsize] * 3,
                        pose=[*sentinel.tolist(), 1, 0, 0, 0],
                    )
                ]
            )
        )

        def query():
            ks = solver.kinematics.compute_kinematics(native_state)
            buffer = CollisionBuffer.from_shape(
                ks.robot_spheres.shape, bindings.device_config_cls(device="cuda")
            )
            return (
                solver.scene_collision_checker.get_sphere_distance(
                    ks,
                    buffer,
                    torch.ones(1, device="cuda"),
                    torch.zeros(1, device="cuda"),
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .copy()
            )

        with_payload = query()
        backend._detach_motion_attachment(solver)
        without_payload = query()
        assert np.max(with_payload[indices]) > 0
        assert np.max(with_payload[~indices]) <= 0 and np.max(without_payload) <= 0
        assert torch.all(params.get_link_spheres("attached_object")[:, 3] < 0)
        report["sentinel"] = {
            "passed": True,
            "probe_halfsize_m": halfsize,
            "position_robot_root_m": sentinel.tolist(),
            "attached_payload_cost_max": float(with_payload[indices].max()),
            "arm_cost_max": float(with_payload[~indices].max()),
            "detached_cost_max": float(without_payload.max()),
            "detached_slots_all_negative": True,
            "scope": "real planner world-collision kernel; not trajectory feasibility or self-collision acceptance",
        }
        # Same mesh/config, different native state lifecycle: cache + reset/update checks.
        torch.cuda.synchronize()
        cached_started = time.perf_counter()
        again = backend._sync_motion_attachment(
            bindings, solver, joint_state, native_state, root
        )
        torch.cuda.synchronize()
        report["cached_reattach_including_fk_time_s"] = (
            time.perf_counter() - cached_started
        )
        report["reattach_cache_hit"] = again.cache_hit
        assert again.cache_hit and again.actual_count == len(original)
        if getattr(args, "extended_native_checks", False):
            from .native_validation import verify_lifecycle_and_models, verify_two_environments
            report["model_switch"] = verify_lifecycle_and_models(
                backend, solver, native_state, original, context, bindings, root, query, indices, sentinel,
            )
            report["multi_env"] = verify_two_environments(solver.kinematics, native_state, original)
        backend._detach_motion_attachment(solver)
        report["env_idx"] = 0
        report["multi_env_verified"] = bool(report.get("multi_env", {}).get("passed"))
        dump(args.output / (args.method + "_native.json"), report)
        # Display the actual native readback, NOT a second transform of object spheres.
        stage = Usd.Stage.Open(str(path))
        preview.set_matrix(
            stage.GetPrimAtPath("/World/CollisionSpheres"), Gf.Matrix4d(1)
        )
        for i, sphere in enumerate(native_world):
            prim = stage.GetPrimAtPath(f"/World/CollisionSpheres/sphere_{i:03d}")
            UsdGeom.Xformable(prim).ClearXformOpOrder()
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*sphere[:3]))
            UsdGeom.Sphere(prim).GetRadiusAttr().Set(float(sphere[3]))
        stage.GetRootLayer().Save()
        evidence["sphere_visual_source"] = (
            "native attachment slot readback + native FK (env0)"
        )
        evidence["native_report"] = str(args.output / (args.method + "_native.json"))
        evidence["native_collision_check_executed"] = True
        dump(evidence_path, evidence)
        print(
            json.dumps(
                {
                    "native_passed": True,
                    "closure_m": closure,
                    "sentinel": report["sentinel"],
                }
            ),
            flush=True,
        )
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("manual_frozen", "global", "auto_geometry", "convex_surface"),
        default="auto_geometry",
    )
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=config_path("mug/1e_mug_auto_geometry.json"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--diagnostic-budget", type=int, default=64)
    parser.add_argument("--extended-native-checks", action="store_true", help="Also verify fewer-slot/model-switch sentinel, world lifecycle, and two-env native FK")
    parser.add_argument(
        "--view", choices=("close", "robot", "top", "side"), default="close"
    )
    parser.add_argument("--hide-visual", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--open-existing", action="store_true")
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Explicitly visual-only; not native acceptance",
    )
    parser.add_argument(
        "--write-run-config",
        action="store_true",
        help="Write an isolated 1e full-task sphere experiment YAML",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == preview.DEFAULT_OUTPUT or args.output == FROZEN.parent:
        parser.error(
            "Comparison output must not overwrite the approved manual directory"
        )
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    args.device = "cuda"
    args.structure_config = preview.DEFAULT_STRUCTURE_CONFIG
    lib = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != lib:
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        os.execvpe(sys.executable, [sys.executable, "-m", "asset_collision_spheres.adapters.workspace.mug_auto", *sys.argv[1:]], env)
    path = args.output / (args.method + ".usda")
    if args.method != "manual_frozen" and "geometry_policy" in json.loads(args.geometry_config.read_text()):
        if args.output.resolve() == OUTPUT.resolve():
            parser.error("geometry_policy requires a separate --output directory; keep historical mug results")
        if path.exists() and not args.open_existing:
            parser.error("Geometry preview already exists; use --open-existing or a new output directory")
    if not args.open_existing:
        path = preview.build(args, fit_override=fit)
        evidence_path = path.with_suffix(".json")
        evidence = json.loads(evidence_path.read_text())
        if args.method == "convex_surface" or evidence["fit"].get("geometry_policy_selected") == "convex_hull":
            from pxr import Gf, Usd, UsdGeom
            from ...preview.convex import add_hull_layers
            stage = Usd.Stage.Open(str(path))
            layer = UsdGeom.Xform.Define(stage, "/World/ConvexDebug")
            preview.set_matrix(layer.GetPrim(), Gf.Matrix4d(evidence["world_from_mug_column_matrix"]).GetTranspose())
            add_hull_layers(stage, evidence["fit"], root="/World/ConvexDebug")
            for name in ("HullSolid", "Patches", "Proxy", "Sites", "HullFeatures"):
                UsdGeom.Imageable(stage.GetPrimAtPath("/World/ConvexDebug/" + name)).MakeInvisible()
            stage.GetRootLayer().Save()
        evidence.update(
            scope=__doc__,
            auto_geometry_generation_used=args.method in {"global", "auto_geometry"},
            convex_surface_generation_used=args.method == "convex_surface",
            automatic_local_balancing_enabled=args.method == "auto_geometry",
            native_collision_check_executed=False,
        )
        if "geometry_routing" in evidence["fit"]:
            evidence["automatic_local_balancing_enabled"] = False
            evidence["geometry_routing"] = evidence["fit"]["geometry_routing"]
        dump(evidence_path, evidence)
        if not args.skip_native:
            native_check(args, path)
    if not path.is_file():
        parser.error("Preview missing: build first without --open-existing")
    if args.write_run_config:
        native_report = args.output / (args.method + "_native.json")
        saved_evidence = json.loads(path.with_suffix(".json").read_text())
        if (
            not native_report.is_file()
            or not json.loads(native_report.read_text())["sentinel"]["passed"]
            or "native" not in saved_evidence.get("sphere_visual_source", "")
        ):
            parser.error(
                "Verify native attachment first before exporting the full-task experiment"
            )
        if args.method != "manual_frozen":
            saved_native = json.loads(native_report.read_text())
            # New scale-aware/null and partial system-policy inputs resolve to
            # concrete settings. Verify the exact raw request, not raw-vs-resolved.
            checked = saved_native.get("requested_mesh_config", saved_native["mesh_diagnostics"]["config"])
            expected = {
                **json.loads(args.geometry_config.read_text()),
                "mode": args.method,
            }
            if any(checked.get(key) != value for key, value in expected.items()):
                parser.error(
                    "Geometry config changed since native verification; rebuild first"
                )
        authored = write_runtime_config(args)
        runtime = materialize_offline_run(authored)
        print("Experiment run config: " + str(authored), flush=True)
        print("Offline resolved run: " + str(runtime), flush=True)
    if not args.build_only:
        args.camera_path = "/World/Camera" + args.view.capitalize()
        args.hidden_paths = ["/World/MugAtAttach"] if args.hide_visual else []
        args.preview_legend = f"1e {args.method} | purple=initial mesh; colors=patches, NOT semantic parts; static posture, see JSON for native verification"
        if args.method == "convex_surface":
            args.preview_legend = "1e CONVEX V3 | native FK spheres; gold=rim, other colors=sampling sources; cyan=hull, purple=initial mesh | static posture"
        if "geometry_policy" in json.loads(args.geometry_config.read_text()):
            selected = json.loads(path.with_suffix(".json").read_text())["fit"]["geometry_policy_selected"]
            args.preview_legend = f"1e GEOMETRY V4 | target={selected} | spheres=native FK if verified; purple=initial mesh | static posture"
        preview.open_isaac(path, args)


if __name__ == "__main__":
    main()
