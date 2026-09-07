"""USD drawing and static pose helpers. USD imports are deferred."""
import math

def matrix(pose):
    from pxr import Gf

    x, y, z, w = pose["quat_xyzw"]
    result = Gf.Matrix4d(1)
    result.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    result.SetTranslateOnly(Gf.Vec3d(*pose["xyz_m"]))
    return result



def set_matrix(prim, value):
    from pxr import UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp(opSuffix="staticPreview").Set(value)



def mesh_data(stage, collision_only):
    import numpy as np
    import trimesh
    from pxr import Gf, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
    parts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        is_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if is_collision != collision_only:
            continue
        mesh = UsdGeom.Mesh(prim)
        xf = cache.GetLocalToWorldTransform(prim) * root_inv
        vertices = np.asarray(
            [
                tuple(xf.Transform(Gf.Vec3d(*map(float, p))))
                for p in mesh.GetPointsAttr().Get()
            ]
        )
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces, offset = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset : offset + count]
            faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
            offset += count
        parts.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    if not parts:
        raise RuntimeError(
            "Requested mesh is absent; refusing to substitute visual geometry"
        )
    return trimesh.util.concatenate(parts)



def pose_robot(stage, path, root_pose, positions):
    """USD joint-frame FK, including authored mimic relations; no physics stepping."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    robot = stage.GetPrimAtPath(path)
    set_matrix(robot, matrix(root_pose))
    joints = [
        UsdPhysics.Joint(p) for p in Usd.PrimRange(robot) if p.IsA(UsdPhysics.Joint)
    ]
    values = dict(positions)
    for _ in range(len(joints)):
        for j in joints:
            p = j.GetPrim()
            for rel in p.GetRelationships():
                if "MimicJoint" not in rel.GetName() or not rel.GetName().endswith(
                    "referenceJoint"
                ):
                    continue
                targets = rel.GetTargets()
                if not targets:
                    continue
                prefix = rel.GetName().rsplit(":", 1)[0]
                gearing = float(p.GetAttribute(prefix + ":gearing").Get())
                offset = float(p.GetAttribute(prefix + ":offset").Get() or 0)
                # PhysX constraint: q + gearing * reference + offset = 0.
                values[p.GetName()] = (
                    -gearing * values.get(targets[0].name, 0.0) - offset
                )
    worlds, pending = {}, list(joints)

    def frame(pos, quat):
        m = Gf.Matrix4d(1)
        m.SetRotate(Gf.Quatd(quat))
        m.SetTranslateOnly(Gf.Vec3d(pos))
        return m

    while pending:
        remaining = []
        for joint in pending:
            p = joint.GetPrim()
            parent, child = (
                joint.GetBody0Rel().GetTargets(),
                joint.GetBody1Rel().GetTargets(),
            )
            if not child:
                continue
            if parent and str(parent[0]) not in worlds:
                remaining.append(joint)
                continue
            parent_world = worlds[str(parent[0])] if parent else matrix(root_pose)
            f0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            f1 = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            motion = Gf.Matrix4d(1)
            q = values.get(p.GetName(), 0.0)
            if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
                axis = str(p.GetAttribute("physics:axis").Get())
                v = {
                    "X": Gf.Vec3d(1, 0, 0),
                    "Y": Gf.Vec3d(0, 1, 0),
                    "Z": Gf.Vec3d(0, 0, 1),
                }[axis]
                if p.IsA(UsdPhysics.RevoluteJoint):
                    motion.SetRotate(Gf.Rotation(v, math.degrees(q)))
                else:
                    motion.SetTranslate(v * q)
            elif not p.IsA(UsdPhysics.FixedJoint):
                raise RuntimeError(f"Unsupported static FK joint: {p.GetPath()}")
            world = f1.GetInverse() * motion * f0 * parent_world
            worlds[str(child[0])] = world
            child_prim = stage.GetPrimAtPath(child[0])
            parent_xf = UsdGeom.XformCache().GetLocalToWorldTransform(
                child_prim.GetParent()
            )
            set_matrix(child_prim, world * parent_xf.GetInverse())
        if len(remaining) == len(pending):
            raise RuntimeError("Robot joint graph is not a supported tree")
        pending = remaining
    return worlds



def material(stage, path, color, opacity):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat



def bind(prim, mat):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )



def wire_mesh(stage, path, mesh, color, width=0.0006):
    from pxr import Gf, UsdGeom

    edges = mesh.edges_unique
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2] * len(edges))
    curve.CreatePointsAttr(
        [Gf.Vec3f(*map(float, mesh.vertices[i])) for edge in edges for i in edge]
    )
    curve.CreateWidthsAttr([width])
    curve.SetWidthsInterpolation("constant")
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])

