"""Isolated real CUDA validation. Never changes task physics or backend source."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import time


def verify_lifecycle_and_models(
    backend, solver, state, original, context, bindings, root, query, indices, sentinel
):
    import numpy as np
    import torch
    from fastsim_plugin_mission.task.unisolver.core import Ok, RobotAttachmentState

    manager = solver.attachment_manager
    params = manager.kinematics_params
    full = params.get_link_spheres("attached_object").clone()
    actual = len(original)
    original_cost = query()
    world_spheres = (
        solver.kinematics.compute_kinematics(state)
        .robot_spheres.detach()
        .cpu()
        .numpy()
        .reshape(-1, 4)[indices][:actual]
    )
    # A deliberately reduced diagnostic model removes every sphere that could
    # hit this sentinel. This is not an approved output or a collision ignore.
    clearance = (
        np.linalg.norm(world_spheres[:, :3] - sentinel, axis=1) - world_spheres[:, 3]
    )
    keep = np.flatnonzero(clearance > 0.003)
    keep = keep[: max(1, actual // 2)]
    if not len(keep) or len(keep) >= actual:
        raise RuntimeError("Cannot construct an isolated reduced-model sentinel test")
    manager.update(
        full[torch.as_tensor(keep, device=full.device)],
        state,
        link_name="attached_object",
    )
    reduced_cost = query()
    written = params.get_link_spheres("attached_object")
    assert torch.all(written[len(keep) :, 3] < 0)
    assert float(original_cost[indices].max()) > 0 and float(reduced_cost.max()) <= 0
    manager.update(full[:actual], state, link_name="attached_object")
    restored_cost = query()
    assert float(restored_cost[indices].max()) > 0
    # Measure actual kernel query cost separately from mesh fitting / FK sync.
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(20):
        query()
    torch.cuda.synchronize()
    query_ms = (time.perf_counter() - started) * 1000 / 20
    # The Mission adapter excludes the held object from the materialized world;
    # it re-enters the world when the attachment context is removed.
    assert "mug" in backend._moving_object_ids(context.collision_world, root)
    from fastsim_plugin_mission.task.unisolver.core import CollisionPolicy

    attached_world = backend._materialized_world(
        bindings,
        context.collision_world,
        root,
        CollisionPolicy(),
        include_geometry_facts=True,
    )
    assert not any("mug" in name for name in attached_world.obstacle_ids)
    detached = replace(
        context,
        revision=context.revision + 1,
        attachments=RobotAttachmentState("g2", context.attachments.revision + 1, ()),
    )
    assert backend.apply_arm_planning_context(detached) == Ok(None)
    assert "mug" not in backend._moving_object_ids(detached.collision_world, root)
    # Materialization is read-only w.r.t. Isaac and uses the same real mesh.
    materialized = backend._materialized_world(
        bindings,
        detached.collision_world,
        root,
        CollisionPolicy(),
        include_geometry_facts=True,
    )
    assert any("mug" in name for name in materialized.obstacle_ids)
    return {
        "passed": True,
        "original_count": actual,
        "reduced_count": len(keep),
        "capacity": len(full),
        "unused_slots_negative_after_reduction": True,
        "full_payload_cost": float(original_cost[indices].max()),
        "reduced_model_cost": float(reduced_cost.max()),
        "restored_payload_cost": float(restored_cost[indices].max()),
        "world_excluded_while_attached_and_materialized_after_detach": True,
        "detached_world_obstacle_ids": list(materialized.obstacle_ids),
        "mean_fk_kernel_readback_query_ms": query_ms,
        "scope": "same real planner: original -> deliberately reduced diagnostic subset -> restored; not an optimized reduced model or trajectory benchmark",
    }


def verify_two_environments(source_model, state, object_spheres):
    """Separate native G2 model, two grasp poses and two object-relative poses."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from curobo._src.collision.attachment_manager import AttachmentManager
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.pose import Pose

    cfg = deepcopy(source_model.config)
    params = cfg.kinematics_config
    params.link_spheres = params.link_spheres[:1].repeat(2, 1, 1)
    params.reference_link_spheres = params.reference_link_spheres[:1].repeat(2, 1, 1)
    model = Kinematics(cfg)
    manager = AttachmentManager(kinematics=model, device_cfg=model.device_cfg)
    q = state.position.reshape(1, -1).repeat(2, 1)
    q[1, -1] += 0.1
    joints = JointState.from_position(q, joint_names=model.joint_names)
    options = dict(device=q.device, dtype=q.dtype)
    poses = Pose(
        position=torch.tensor([[0.6, -0.2, 0.5], [0.4, 0.3, 0.6]], **options),
        quaternion=torch.tensor(
            [[1, 0, 0, 0], [0.9238795325, 0, 0, 0.3826834324]], **options
        ),
    )
    spheres = torch.as_tensor(np.asarray(object_spheres), **options)
    idx = params.get_sphere_index_from_link_name("attached_object")
    try:
        manager.update(spheres, joints, world_objects_pose_offset=poses)
        local = params.link_spheres[:, idx, :].clone()
        assert not torch.allclose(
            local[0, : len(spheres), :3], local[1, : len(spheres), :3]
        )
        fk = model.compute_kinematics(
            joints, idxs_env=torch.arange(2, device=q.device, dtype=torch.int32)
        )
        actual = fk.robot_spheres.reshape(2, -1, 4)[:, idx[: len(spheres)], :]
        rotation = Rotation.from_quat(
            np.roll(poses.quaternion.cpu().numpy(), -1, axis=1)
        ).as_matrix()
        expected_np = (
            np.einsum("eij,nj->eni", rotation, np.asarray(object_spheres)[:, :3])
            + poses.position.cpu().numpy()[:, None]
        )
        expected = torch.as_tensor(expected_np, **options)
        error = torch.linalg.vector_norm(actual[:, :, :3] - expected, dim=-1)
        assert float(error.max()) < 2e-5
        assert torch.allclose(
            actual[:, :, 3], spheres[:, 3].expand(2, -1), atol=1e-7, rtol=0
        )
        manager.update(
            spheres[: max(1, len(spheres) // 2)],
            joints,
            world_objects_pose_offset=poses,
        )
        assert torch.all(
            params.link_spheres[:, idx[max(1, len(spheres) // 2) :], 3] < 0
        )
        manager.detach()
        assert torch.all(params.link_spheres[:, idx, 3] < 0)
        return {
            "passed": True,
            "env_count": 2,
            "actual_spheres_per_env": len(spheres),
            "center_error_m_per_env": error.max(dim=1).values.cpu().tolist(),
            "per_env_local_centers_different": True,
            "reduction_and_detach_slots_negative": True,
            "scope": "real native G2 per-env attachment + indexed CUDA FK; no two-environment Isaac physics or batched trajectory run",
        }
    finally:
        manager.detach()
