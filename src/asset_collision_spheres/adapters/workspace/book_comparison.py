"""Aligned 1g book comparison using the real collider and matched budgets.

Static reconstruction from an attach log, NOT a synchronized measured robot
state or a native planner readback. The purple outline remains at initial pose.
Only preview layers are written; 1g/1e runtime configs and mug results stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from ...paths import workspace_root, output_root, config_path

from .book import (
    PROJECT,
    ROOT,
    bind,
    material,
    matrix,
    mesh_data,
    open_isaac,
    pose_robot,
    set_matrix,
    source_state,
    wire_mesh,
)

LEGACY_OUTPUT = PROJECT / "sphere_preview_box"
DEFAULT_OUTPUT = output_root() / "book/sphere_preview_box"
def load_generator(filename):
    from importlib import import_module
    names = {"mesh_region_spheres": "mesh_regions", "solid_box_regions": "solid_box", "attachment_spheres": "analytic"}
    return import_module("asset_collision_spheres.algorithms." + names[filename])


def aligned_book_asset(source_path, output):
    """Visual-only USD overlay: preserve every original collision/physics opinion."""
    import numpy as np
    from pxr import Usd, UsdGeom

    source_path = Path(source_path).resolve()
    original = Usd.Stage.Open(str(source_path))
    before_visual, before_collision = (
        mesh_data(original, False),
        mesh_data(original, True),
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "book_visual_aligned.usda"
    stage = (
        Usd.Stage.CreateNew(str(path))
        if not path.exists()
        else Usd.Stage.Open(str(path))
    )
    # This generated layer contains no independent authored geometry/physics.
    stage.GetRootLayer().Clear()
    stage.GetRootLayer().subLayerPaths = [str(source_path)]
    visual = original.GetPrimAtPath("/Asset/VisualSource")
    if not original.GetPrimAtPath("/Asset/VisualSource/Source"):
        if not visual:
            raise RuntimeError(
                "Book no longer uses the known source-reference structure"
            )
        refs = visual.GetMetadata("references").GetAddedOrExplicitItems()
        if len(refs) != 1 or not Path(refs[0].assetPath).is_absolute():
            raise RuntimeError(
                "Unexpected visual reference; refusing an inferred repair"
            )
        transform = UsdGeom.Xformable(visual).GetTransformOp().Get()
        # Same composition correction as the repaired converter: normalize on
        # the parent, retain the source's own scale on a separate referenced child.
        parent = UsdGeom.Xform(stage.OverridePrim("/Asset/VisualSource"))
        parent.GetPrim().GetReferences().SetReferences([])
        parent.ClearXformOpOrder()
        parent.AddTransformOp(opSuffix="alignmentRepair").Set(transform)
        child = UsdGeom.Xform.Define(stage, "/Asset/VisualSource/Source")
        child.GetPrim().GetReferences().SetReferences(list(refs))
        child.SetResetXformStack(False)
    repaired_collision, repaired_visual = (
        mesh_data(stage, True),
        mesh_data(stage, False),
    )
    np.testing.assert_array_equal(
        before_collision.vertices, repaired_collision.vertices
    )
    np.testing.assert_array_equal(before_collision.faces, repaired_collision.faces)
    error = float(np.max(np.abs(repaired_visual.bounds - repaired_collision.bounds)))
    if error > 2e-6:
        raise RuntimeError(
            f"Repaired visual/collision bounds still differ by {error} m"
        )
    # A reference with no explicit prim path looks up defaultPrim on the target
    # root layer; an inherited sublayer opinion alone is insufficient here.
    stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    stage.GetRootLayer().Save()
    return path, {
        "source_asset": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "overlay_asset": str(path),
        "old_center_offset_m": (
            before_collision.bounds.mean(0) - before_visual.bounds.mean(0)
        ).tolist(),
        "new_bounds_error_m": error,
        "collision_vertices_unchanged": True,
        "collision_faces_unchanged": True,
        "scope": "visual-only preview sublayer; original normalized asset and runtime component URI not changed",
    }


def fit_sets(mesh, args):
    import numpy as np
    import torch
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
    from curobo.types import DeviceCfg

    mesh_module = load_generator("mesh_region_spheres")
    family_module = load_generator("solid_box_regions")
    profile = family_module.make_solid_box_region_profile(
        mesh, budget=args.budget, max_outward_offset_m=args.max_outward, seed=0
    )
    improved = mesh_module.generate_mesh_region_spheres(mesh, profile.config)
    np.random.seed(0)
    torch.manual_seed(0)
    baseline = fit_spheres_to_mesh(
        mesh,
        num_spheres=args.budget,
        fit_type=SphereFitType.MORPHIT,
        compute_metrics=True,
        device_cfg=DeviceCfg(device=args.device),
    )
    if baseline.debug_info.get("containment_enforced"):
        raise RuntimeError("Excluded containment debug behavior is enabled")
    original_spheres = np.c_[
        baseline.centers.detach().cpu().numpy(),
        baseline.radii.detach().cpu().numpy().reshape(-1),
    ]
    old_module = load_generator("attachment_spheres")
    old_request = old_module.parse_structure_aware_request(
        {
            "shape_hint": "solid_box",
            "dimensions_m": mesh.extents.tolist(),
            "sphere_budget": args.budget,
            "task_profile": "narrow_passage",
            "max_outward_offset_m": 0.001,
            "validation_probe_radius_m": 0.006,
        },
        default_budget=args.budget,
        default_seed=0,
    )
    legacy = old_module.generate_structure_aware_spheres(
        old_request, reference_bounds_m=tuple(map(tuple, mesh.bounds))
    )
    sets = {
        "baseline": original_spheres,
        "structure_aware": improved.spheres,
        "legacy_box": np.array([[*s.center_m, s.radius_m] for s in legacy.spheres]),
    }
    reports = {}
    thin = profile.facts["thin_axis_index"]
    for name, spheres in sets.items():
        if (
            len(spheres) > args.budget
            or not np.isfinite(spheres).all()
            or np.any(spheres[:, 3] <= 0)
        ):
            raise RuntimeError("Invalid fitter output or exceeded requested budget")
        reports[name] = mesh_module.evaluate_mesh_spheres(mesh, spheres, profile.config)
        reports[name].update(
            {
                "actual_count": len(spheres),
                "radius_range_m": [
                    float(spheres[:, 3].min()),
                    float(spheres[:, 3].max()),
                ],
                "sphere_union_thickness_bound_m": float(
                    np.max(spheres[:, thin] + spheres[:, 3])
                    - np.min(spheres[:, thin] - spheres[:, 3])
                ),
            }
        )
    return {k: v.tolist() for k, v in sets.items()}, {
        "family": profile.facts,
        "region_config": profile.config,
        "improved_generator": improved.diagnostics,
        "baseline_debug": baseline.debug_info,
        "legacy_algorithm_version": legacy.algorithm_version,
        "legacy_analytic_validation_status": legacy.validation.status.value,
        "independent_mesh_holdouts": reports,
        "budget": args.budget,
        "native_attachment_executed": False,
        "comparison_note": "Same full collision mesh, same requested budget, same independent probe seed; not historical native tensor readback",
    }


def build(args):
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    state = source_state()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset, alignment = aligned_book_asset(state["book_asset"], output / "assets")
    source = Usd.Stage.Open(str(asset))
    collision, visual = mesh_data(source, True), mesh_data(source, False)
    sets, evidence = fit_sets(collision, args)
    path = output / f"{args.method}_{args.budget}.usda"
    stage = (
        Usd.Stage.CreateNew(str(path))
        if not path.exists()
        else Usd.Stage.Open(str(path))
    )
    if stage.GetPrimAtPath("/World"):
        stage.RemovePrim("/World")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1100)
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    robot.GetReferences().AddReference(str(state["robot_asset"]))
    worlds = pose_robot(
        stage,
        "/World/Robot",
        state["run"]["scenario"]["scene"]["robots"]["g2"]["pose"],
        state["joints"],
    )
    anchor = next(
        (m for p, m in worlds.items() if p.endswith("/left_gripper_center")), None
    )
    book = UsdGeom.Xform.Define(stage, "/World/BookAtAttach").GetPrim()
    book.GetReferences().AddReference(str(asset))
    if not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(book)):
        raise RuntimeError("Preview book reference did not compose any mesh")
    book_world = matrix(state["book_pose"])
    set_matrix(book, book_world)
    bind(book, material(stage, "/World/Materials/Book", (0.72, 0.76, 0.8), 0.2))
    for prim in Usd.PrimRange(book):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(prim).MakeInvisible()
    reference = UsdGeom.Xform.Define(stage, "/World/CollisionReference")
    set_matrix(reference.GetPrim(), book_world)
    wire_mesh(
        stage, "/World/CollisionReference/Mesh", collision, (0.08, 0.85, 0.7), 0.0001
    )
    UsdGeom.Imageable(reference).MakeInvisible()
    original = UsdGeom.Xform.Define(stage, "/World/BookInitialPosition")
    set_matrix(
        original.GetPrim(),
        matrix(state["run"]["scenario"]["scene"]["objects"]["book_a"]["pose"]),
    )
    wire_mesh(
        stage, "/World/BookInitialPosition/Outline", visual, (0.8, 0.25, 0.85), 0.00012
    )
    root = UsdGeom.Xform.Define(stage, "/World/CollisionSpheres")
    set_matrix(root.GetPrim(), book_world)
    colors = {
        "baseline": (1.0, 0.42, 0.05),
        "structure_aware": (0.1, 0.65, 1.0),
        "legacy_box": (0.65, 0.8, 0.2),
    }
    mat = material(
        stage,
        "/World/Materials/Spheres",
        colors.get(args.method, (0.5, 0.5, 0.5)),
        0.85,
    )
    spheres = [] if args.method == "reference" else sets[args.method]
    for i, (x, y, z, r) in enumerate(spheres):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/CollisionSpheres/sphere_{i:03d}")
        sphere.CreateRadiusAttr(r)
        sphere.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        bind(sphere.GetPrim(), mat)
        if args.method == "structure_aware":
            sphere.GetPrim().SetCustomDataByKey(
                "structureRegion", evidence["improved_generator"]["sphere_regions"][i]
            )
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    center = book_world.Transform(Gf.Vec3d(*map(float, collision.bounds.mean(0))))
    offsets = {
        "Close": [0.28, -0.42, 0.34],
        "Top": [0.0, 0.0, 0.7],
        "Side": [0.35, -0.4, 0.08],
        "Robot": [2.0, -3.0, 1.6],
    }
    for name, offset in offsets.items():
        camera = UsdGeom.Camera.Define(stage, "/World/Camera" + name)
        camera_up = (
            book_world.TransformDir(Gf.Vec3d(0, 0, 1))
            if name == "Top"
            else Gf.Vec3d(0, 0, 1)
        )
        set_matrix(
            camera.GetPrim(),
            Gf.Matrix4d()
            .SetLookAt(center + Gf.Vec3d(*offset), center, camera_up)
            .GetInverse(),
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100))
        camera.CreateFocalLengthAttr(32 if name != "Robot" else 24)
    stage.GetRootLayer().Save()
    evidence.update(
        {
            "scope": __doc__,
            "usd": str(path),
            "method_shown": args.method,
            "alignment": alignment,
            "source_run": str(PROJECT / "run.yaml"),
            "source_log": state["log"],
            "source_joint_targets": state["outcomes"],
            "book_pose": state["book_pose"],
            "initial_pose": state["run"]["scenario"]["scene"]["objects"]["book_a"][
                "pose"
            ],
            "robot_asset": str(state["robot_asset"]),
            "robot_pose_note": (
                "Current robot asset plus historical joint targets, NOT a synchronized "
                "measured snapshot. Missing historical anchor is not treated as zero error."
            ),
            "robot_anchor_error_m": float(
                (
                    anchor.ExtractTranslation()
                    - Gf.Vec3d(*state["parent_pose"]["xyz_m"])
                ).GetLength()
            )
            if anchor is not None
            else None,
            "spheres_object_local_m": sets,
        }
    )
    path.with_suffix(".json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "usd": str(path),
                "alignment": alignment,
                "reports": evidence["independent_mesh_holdouts"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("baseline", "structure_aware", "legacy_box", "reference"),
        default="structure_aware",
    )
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--max-outward", type=float, default=0.006)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--view", choices=("close", "top", "side", "robot"), default="close"
    )
    parser.add_argument("--hide-visual", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--open-existing", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.budget < 14 or args.budget > 256:
        parser.error("Matched legacy/updated comparison requires budget in [14, 256]")
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    library_dir = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != library_dir:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = library_dir + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        os.execvpe(sys.executable, [sys.executable, "-m", "asset_collision_spheres.adapters.workspace.book_comparison", *sys.argv[1:]], env)
    path = (
        args.output.resolve() / f"{args.method}_{args.budget}.usda"
        if args.open_existing
        else build(args)
    )
    if not path.is_file():
        parser.error(f"Missing preview: {path}")
    if not args.build_only:
        args.camera_path = "/World/Camera" + args.view.capitalize()
        args.hidden_paths = ["/World/BookAtAttach"] if args.hide_visual else []
        args.preview_legend = "1g BOOK | orange=public MORPHIT; blue=updated flat-box mesh regions; purple=initial book. Static, NOT native readback."
        open_isaac(path, args)


if __name__ == "__main__":
    main()
