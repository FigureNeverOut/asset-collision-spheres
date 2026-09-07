"""Optional OpenUSD loader; applies authored transform, explicit units and Z-up rotation.

Output is the original static preview frame, not automatically an attachment frame.
"""
import hashlib
from pathlib import Path

def load_material_mesh(preset):
    import numpy as np
    import trimesh
    from .mesh_attachment import (
        check_material_mesh,
    )
    from pxr import Gf, Usd, UsdGeom

    source = preset["source"]
    path = Path(source["usd_path"]).expanduser().resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open source USD: {path}")
    prim = stage.GetPrimAtPath(source["mesh_prim"])
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError("mesh_prim must explicitly select one full-material Mesh")
    geometry = UsdGeom.Mesh(prim)
    counts = np.asarray(geometry.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3) or geometry.GetHoleIndicesAttr().Get():
        raise ValueError("Preview requires a triangulated mesh without USD hole faces")
    points = np.asarray(geometry.GetPointsAttr().Get(), dtype=float)
    faces = np.asarray(geometry.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    vertices = np.asarray([xf.Transform(Gf.Vec3d(*p)) for p in points])
    scale = float(source["unit_scale_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("unit_scale_m must be finite and positive")
    vertices *= scale
    # Optional independent dimension guard, measured before display-axis rotation.
    if "expected_extents_m" in source and not np.allclose(
        np.ptp(vertices, axis=0), source["expected_extents_m"], rtol=0.005, atol=1e-5
    ):
        raise ValueError("Mesh extents disagree with explicitly supplied physical size")
    up = source.get("up_axis", str(UsdGeom.GetStageUpAxis(stage))).upper()
    if up == "Y":
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    elif up == "Z":
        rotation = np.eye(3)
    else:
        raise ValueError("Supported source up_axis values: Y, Z")
    vertices = vertices @ rotation.T
    # Preserve outward winding under authored reflections or left-handed topology.
    if (geometry.GetOrientationAttr().Get() == "leftHanded") != (
        xf.GetDeterminant() < 0
    ):
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    check_material_mesh(mesh)
    return mesh, {
        "usd_path": str(path),
        "usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mesh_prim": str(prim.GetPath()),
        "authored_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "effective_unit_scale_m": scale,
        "source_up_axis": up,
        "source_stage_to_preview_rotation": rotation.tolist(),
        "source_mesh_to_stage_row_matrix": np.asarray(xf).tolist(),
        "bounds_preview_m": mesh.bounds.tolist(),
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume_m3": float(mesh.volume),
        "geometry_repair_applied": False,
        "annotation_used_for_generation": False,
        "reference": "full source mesh, not a PhysX convex approximation",
    }

