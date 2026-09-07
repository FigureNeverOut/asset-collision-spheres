"""Optional OpenUSD loader; applies authored transform, explicit units and Z-up rotation.

Output is the original static preview frame, not automatically an attachment frame.
"""
import hashlib
from pathlib import Path

def load_material_mesh(preset, *, require_material=True, require_triangles=False):
    import numpy as np
    import trimesh
    from ..geometry.validation import (
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
    if require_triangles and not np.all(counts == 3):
        raise ValueError("Geometry routing/original_surface requires authored triangles; debug polygon fans are not a physical surface")
    if require_material and (not np.all(counts == 3) or geometry.GetHoleIndicesAttr().Get()):
        raise ValueError("Preview requires a triangulated mesh without USD hole faces")
    points = np.asarray(geometry.GetPointsAttr().Get(), dtype=float)
    indices = np.asarray(geometry.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if require_material:
        faces = indices.reshape(-1, 3)
    else:
        # Convex generation consumes vertices, not winding / hole topology.
        # Fan triangulation here is debug-display only; declared hole faces stay
        # absent. Unit, authored transform and up-axis handling below is shared.
        if np.any(counts < 3) or int(counts.sum()) != len(indices):
            raise ValueError("Invalid USD polygon topology")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
            raise ValueError("Invalid USD vertex indices")
        holes = set(geometry.GetHoleIndicesAttr().Get() or [])
        triangles, offset = [], 0
        for face_id, count in enumerate(counts):
            polygon = indices[offset:offset + count]
            if face_id not in holes:
                triangles.extend((polygon[0], polygon[i], polygon[i+1]) for i in range(1, count - 1))
            offset += count
        faces = np.asarray(triangles, dtype=int).reshape(-1, 3)
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
    if require_material:
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
        "volume_m3": (float(mesh.volume) if require_material or (mesh.is_watertight and mesh.is_winding_consistent) else None),
        "geometry_repair_applied": False,
        "annotation_used_for_generation": False,
        "reference": "full source mesh, not a PhysX convex approximation",
    }
