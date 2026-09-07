import numpy as np
import pytest
import trimesh

from asset_collision_spheres.adapters.fastsim import (
    fit_mesh_attachment,
    prepare_request,
)
from asset_collision_spheres.algorithms.geometry_policy import normalize
from asset_collision_spheres.api import (
    generate_convex_surface_spheres,
    generate_geometry_spheres,
)


def case(name):
    import runpy
    from pathlib import Path

    return runpy.run_path(
        str(Path(__file__).parents[1] / "examples/geometry_cases.py")
    )["synthetic_case"](name)


def test_forced_hull_is_exact_legacy_delegation():
    mesh = trimesh.creation.box(extents=[0.3, 0.2, 0.1])
    old = generate_convex_surface_spheres(mesh, {"max_spheres": 32})
    new = generate_geometry_spheres(
        mesh, {"geometry_policy": "convex_hull", "max_spheres": 32}
    )
    np.testing.assert_array_equal(old.spheres, new.spheres)


def test_forced_original_adapter():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    config, digest = prepare_request(
        {"geometry_policy": "original_surface", "max_spheres": 32}, capacity=64, seed=5
    )
    result = fit_mesh_attachment(mesh, config)
    assert result.spheres.shape == (32, 4)
    assert result.diagnostics["surface_site_max_error_m"] < 1e-8
    assert (
        digest
        != prepare_request(
            {"geometry_policy": "convex_hull", "max_spheres": 32}, capacity=64, seed=5
        )[1]
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"max_spheres": 65},
        {"max_spheres": True},
        {"geometry_policy": "cup"},
        {"geometry_policy": "auto", "mode": "convex_surface"},
        {"sphere_budget": 32},
        {"radius": 0.2},
    ],
)
def test_invalid_config(raw):
    with pytest.raises(ValueError):
        normalize(raw)


@pytest.mark.parametrize(
    "extents,expected", [([0.2, 0.2, 0.2], 56), ([0.4, 0.2, 0.12], 64)]
)
def test_original_box_grid(extents, expected):
    mesh = trimesh.creation.box(extents=extents)
    result = generate_geometry_spheres(mesh, {"geometry_policy": "original_surface"})
    assert len(result.spheres) == expected
    assert result.diagnostics["sampling_strategy"] == "original_box_grid"
    assert result.regions.count("corner") == 8
    moved = mesh.copy()
    transform = trimesh.transformations.euler_matrix(0.43, -0.68, 0.9)
    transform[:3, 3] = [0.4, -0.8, 0.3]
    moved.apply_transform(transform)
    other = generate_geometry_spheres(moved, {"geometry_policy": "original_surface"})
    from scipy.spatial import cKDTree

    transformed = trimesh.transform_points(result.spheres[:, :3], transform)
    assert cKDTree(transformed).query(other.spheres[:, :3])[0].max() < 1e-8


@pytest.mark.parametrize("name", ["u_solid", "l_solid", "curved_rod", "branched_solid"])
def test_original_connected_geometry(name):
    result = generate_geometry_spheres(
        case(name), {"geometry_policy": "original_surface", "max_spheres": 32}
    )
    d = result.diagnostics
    assert d["sampling_strategy"] == "original_graph_fps"
    assert d["surface_site_max_error_m"] < 1e-8
    assert not d["connectivity_fairness"]["missing_components"]
    if name == "u_solid":
        p = result.spheres[:, :3]
        assert ((p[:, 0] < 0.026) & (p[:, 1] > 0.10)).sum() >= 4
        assert ((p[:, 0] > 0.149) & (p[:, 1] > 0.10)).sum() >= 4


def test_disconnected_close_arms_have_no_air_shortcut():
    first = trimesh.creation.box(extents=[0.01, 0.01, 0.2])
    second = first.copy()
    second.apply_translation([0.012, 0, 0])
    mesh = trimesh.util.concatenate([first, second])
    result = generate_geometry_spheres(
        mesh,
        {
            "geometry_policy": "original_surface",
            "shape_hint": "generic",
            "max_spheres": 32,
        },
    )
    d = result.diagnostics["connectivity_fairness"]
    assert d["component_count"] == 2
    assert min(c["site_count"] for c in d["components"]) >= 12


def test_graph_distance_bypasses_u_air_gap():
    from scipy.sparse.csgraph import dijkstra

    from asset_collision_spheres.algorithms.surface_graph import build_graph

    mesh = case("u_solid")
    graph, nodes, *_ = build_graph(mesh, 5, 256)
    a = np.argmin(np.linalg.norm(nodes - [0, 0.25, 0], axis=1))
    b = np.argmin(np.linalg.norm(nodes - [0.175, 0.25, 0], axis=1))
    assert dijkstra(graph, indices=a)[b] > 2 * np.linalg.norm(nodes[a] - nodes[b])


def test_box_different_triangulations_same_sites():
    from scipy.spatial import cKDTree

    config = {"geometry_policy": "original_surface"}
    a = generate_geometry_spheres(case("solid_box"), config)
    b = generate_geometry_spheres(case("subdivided_box"), config)
    assert cKDTree(a.spheres[:, :3]).query(b.spheres[:, :3])[0].max() < 1e-8


@pytest.mark.parametrize(
    "name",
    [
        "solid_box",
        "sphere",
        "straight_rod",
        "curved_rod",
        "u_solid",
        "l_solid",
        "ring_solid",
        "branched_solid",
        "hollow_bent_tube",
        "narrow_neck",
        "irregular_scan",
        "rounded_box",
    ],
)
def test_router_negative_and_boundary(name):
    from asset_collision_spheres.algorithms.geometry_router import route_geometry

    report = route_geometry(case(name))
    assert report["geometry_policy_selected"] == "original_surface"
    if name in {"irregular_scan", "narrow_neck", "hollow_bent_tube"}:
        assert report["review_required"]


@pytest.mark.parametrize("sections", [4, 64])
def test_router_container_rotation(sections):
    from asset_collision_spheres.algorithms.geometry_router import route_geometry

    mesh = trimesh.creation.revolve(
        [[0, 0], [0.10, 0], [0.10, 0.16], [0.09, 0.16], [0.09, 0.01], [0, 0.01]],
        sections=sections,
    )
    for moved in [False, True]:
        if moved:
            transform = trimesh.transformations.euler_matrix(0.37, -0.63, 1.11)
            transform[:3, 3] = [0.5, -0.3, 1.2]
            mesh.apply_transform(transform)
        report = route_geometry(mesh)
        assert report["geometry_policy_selected"] == "convex_hull"


def test_tiny_capacity_and_duplicate_seams():
    m = case("u_solid")
    soup = trimesh.Trimesh(
        m.triangles.reshape(-1, 3),
        np.arange(len(m.faces) * 3).reshape(-1, 3),
        process=False,
    )
    result = generate_geometry_spheres(
        soup, {"geometry_policy": "original_surface", "max_spheres": 1}
    )
    assert result.spheres.shape == (1, 4)
    assert result.diagnostics["connectivity_fairness"]["component_count"] == 1


def test_opt_in_auto_dispatch_and_backend_capacity():
    from asset_collision_spheres.api import generate_auto_geometry_spheres

    config, _ = prepare_request(
        {"geometry_policy": "auto", "max_spheres": 64}, capacity=16, seed=5
    )
    result = generate_auto_geometry_spheres(case("u_solid"), config)
    assert len(result.spheres) == 16
    assert result.diagnostics["geometry_policy_selected"] == "original_surface"


def test_generic_rotation_and_retriangulation_quality_not_exact_sites():
    config = {"geometry_policy": "original_surface", "max_spheres": 32}
    mesh = case("u_solid")
    original = generate_geometry_spheres(mesh, config)
    q95 = original.diagnostics["surface_uniformity"]["site_distance_m"]["p95"]
    rotated = mesh.copy()
    transform = trimesh.transformations.euler_matrix(0.37, -0.63, 1.11)
    transform[:3, 3] = [0.5, -0.3, 1.2]
    rotated.apply_transform(transform)
    for changed in [mesh.subdivide(), rotated]:
        result = generate_geometry_spheres(changed, config)
        assert result.diagnostics["surface_site_max_error_m"] < 1e-8
        assert (
            result.diagnostics["surface_uniformity"]["site_distance_m"]["p95"]
            < q95 * 1.5
        )
        assert len(result.spheres) == 32


def test_public_heldout_evaluator_routes_without_material_gate():
    from asset_collision_spheres.api import evaluate_auto_geometry_spheres

    config = {"mode": "auto_geometry", "geometry_policy": "auto", "max_spheres": 32}
    mesh = case("u_solid")
    result = generate_geometry_spheres(mesh, config)
    report = evaluate_auto_geometry_spheres(
        mesh, result.spheres, config, sample_count=700
    )
    assert report["target_geometry"] == "original_surface"
    assert report["surface_uniformity"]["sample_count"] == 700
    bad = result.spheres.copy()
    bad[0, 0] += 1.0
    with pytest.raises(ValueError, match="target surface"):
        evaluate_auto_geometry_spheres(mesh, bad, config)


def test_preview_analysis_hull_is_hidden_and_sites_preserve_frame(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from asset_collision_spheres.preview.geometry import write_stage

    mesh = case("u_solid")
    result = generate_geometry_spheres(
        mesh, {"geometry_policy": "auto", "max_spheres": 32}
    )
    path = tmp_path / "v4.usda"
    write_stage(path, mesh, result)
    stage = Usd.Stage.Open(str(path))
    assert (
        stage.GetPrimAtPath("/World").GetCustomDataByKey("geometryPolicySelected")
        == "original_surface"
    )
    assert (
        UsdGeom.Imageable(
            stage.GetPrimAtPath("/World/RoutingEvidence")
        ).ComputeVisibility()
        == "invisible"
    )
    cache = UsdGeom.XformCache()
    for i, sphere in enumerate(result.spheres):
        prim = stage.GetPrimAtPath(f"/World/Spheres/sphere_{i:03d}")
        np.testing.assert_allclose(
            cache.GetLocalToWorldTransform(prim).ExtractTranslation(), sphere[:3]
        )


def test_original_usd_loader_rejects_debug_polygon_fans(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from asset_collision_spheres.loaders.usd import load_material_mesh

    path = tmp_path / "polygon.usda"
    stage = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Mesh.Define(stage, "/Mesh")
    prim.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    prim.CreateFaceVertexCountsAttr([4])
    prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.GetRootLayer().Save()
    preset = {
        "source": {"usd_path": str(path), "mesh_prim": "/Mesh", "unit_scale_m": 1}
    }
    with pytest.raises(ValueError, match="authored triangles"):
        load_material_mesh(preset, require_material=False, require_triangles=True)
