"""Diagnostic-only planar support, material-supported edges and corner anchors."""

import colorsys


def add_features(stage, mesh, report):
    import numpy as np
    from pxr import Gf, UsdGeom
    from .usd_scene import bind, material

    UsdGeom.Xform.Define(stage, "/World/Features")
    for plane in report["planes"]:
        pid = plane["id"]
        color = colorsys.hsv_to_rgb((pid * 0.618034) % 1, 0.6, 0.9)
        mat = material(stage, f"/World/FeatureMaterials/p{pid}", color, 1)
        shape = UsdGeom.Mesh.Define(stage, f"/World/Features/Planes/p{pid}")
        shape.CreatePointsAttr(mesh.vertices.tolist())
        faces = mesh.faces[plane["face_ids"]]
        shape.CreateFaceVertexCountsAttr([3] * len(faces))
        shape.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        shape.CreateSubdivisionSchemeAttr("none")
        bind(shape.GetPrim(), mat)
    # Unclassified triangles remain visible, so incomplete planar detection is
    # not mistaken for holes removed from the original material.
    assigned = np.zeros(len(mesh.faces), bool)
    for plane in report["planes"]:
        assigned[plane["face_ids"]] = True
    remaining = mesh.faces[~assigned]
    if len(remaining):
        shape = UsdGeom.Mesh.Define(stage, "/World/Features/UnclassifiedMaterial")
        shape.CreatePointsAttr(mesh.vertices.tolist())
        shape.CreateFaceVertexCountsAttr([3] * len(remaining))
        shape.CreateFaceVertexIndicesAttr(remaining.ravel().tolist())
        shape.CreateSubdivisionSchemeAttr("none")
        mat = material(
            stage, "/World/FeatureMaterials/Unclassified", (0.4, 0.4, 0.4), 1
        )
        bind(shape.GetPrim(), mat)
    scale = report["scale_m"]
    for edge in report["edge_segments"]:
        curve = UsdGeom.BasisCurves.Define(
            stage, f"/World/Features/Edges/e{edge['id']}"
        )
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(*p) for p in edge["endpoints_m"]])
        curve.CreateWidthsAttr([scale * 0.009])
        curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr([Gf.Vec3f(0.1, 1.0, 1.0)])
    mat = material(stage, "/World/FeatureMaterials/Anchors", (1, 0.1, 0.3), 1)
    for i, point in enumerate(report["corner_anchors_m"]):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Features/Anchors/a{i}")
        sphere.CreateRadiusAttr(scale * 0.014)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*point))
        bind(sphere.GetPrim(), mat)
