"""Convex-only invariants, geometry failures, structured sampling and integration."""

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree
import trimesh

from asset_collision_spheres.algorithms.convex_surface import (
    generate_convex_surface_spheres as generate,
    choose_grid,
    grid_count,
    normalize,
    shared_radius,
)
from asset_collision_spheres.algorithms.hull_structure import convex_hull, analyze
from asset_collision_spheres.algorithms.hull_diagnostics import obstacle_approaches
from asset_collision_spheres.adapters.fastsim import (
    prepare_request,
    fit_mesh_attachment,
)


def cube():
    return trimesh.creation.box(extents=[0.2] * 3)


def test_cube_grid_shared_nodes_and_no_diagonals():
    result = generate(cube())
    d = result.diagnostics
    assert d["sampling_strategy"] == "grid"
    assert d["selection"]["nodes_xyz"] == [4, 4, 4]
    assert len(result.spheres) == 56
    assert {k: result.regions.count(k) for k in set(result.regions)} == {
        "corner": 8,
        "edge": 24,
        "face": 24,
    }
    assert len(d["hull_structure"]["patches"]) == 6
    assert len(d["hull_structure"]["major_edges"]) == 12
    assert d["feature_diagnostics"]["corners"]["sites_supported"] == 8
    assert d["surface_site_max_error_m"] < 1e-10
    assert d["support_error"]["late_contact_m"]["max"] == 0


def test_long_box_allocates_long_axis_more_nodes():
    nodes = choose_grid([0.6, 0.2, 0.1], 64)
    assert nodes[0] > nodes[1] >= nodes[2]
    assert grid_count(nodes) <= 64


@pytest.mark.parametrize("cap", [1, 2, 7, 8, 16, 32, 64])
def test_caps_and_backend(cap):
    request, _ = prepare_request({"mode": "convex_surface"}, capacity=cap, seed=5)
    result = fit_mesh_attachment(cube(), request)
    assert len(result.spheres) <= cap
    assert "sphere_budget" not in request
    assert result.diagnostics["effective_max_spheres"] == cap
    assert np.isfinite(result.spheres).all()


@pytest.mark.parametrize(
    "config",
    [
        {"max_spheres": 65},
        {"max_spheres": 0},
        {"max_spheres": True},
        {"max_spheres": None},
        {"max_spheres": 32.5},
        {"radius": 0.5},
        {"sphere_budget": 32},
        {"shape_hint": "bowl"},
        {"seed": -1},
        {"backend_sphere_capacity": 0},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        normalize(config)


def test_open_and_bad_winding_meshes_have_same_convex_result():
    mesh = cube()
    open_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:-2, ::-1], process=False)
    assert not open_mesh.is_watertight
    expected = generate(mesh).spheres
    assert np.array_equal(expected, generate(open_mesh).spheres)


@pytest.mark.parametrize(
    "vertices",
    [
        np.empty((0, 3)),
        np.eye(3),
        np.zeros((4, 3)),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1e-9]]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, np.nan]]),
    ],
)
def test_bad_geometry_fails(vertices):
    mesh = trimesh.Trimesh(vertices=vertices, faces=[], process=False)
    with pytest.raises(ValueError):
        generate(mesh)


def test_rotation_translation_and_subdivision_invariant_box():
    mesh = trimesh.creation.box(extents=[0.31, 0.2, 0.12])
    expected = generate(mesh).spheres
    transform = trimesh.transformations.euler_matrix(0.43, -0.68, 0.9)
    transform[:3, 3] = [0.4, -0.8, 0.3]
    rotated = mesh.copy()
    rotated.apply_transform(transform)
    actual = generate(rotated).spheres
    inverse = (actual[:, :3] - transform[:3, 3]) @ transform[:3, :3]
    assert len(actual) == len(expected)
    assert cKDTree(expected[:, :3]).query(inverse)[0].max() < 1e-9
    assert np.allclose(actual[:, 3], expected[:, 3])
    subdivided = generate(mesh.subdivide()).spheres
    assert cKDTree(expected[:, :3]).query(subdivided[:, :3])[0].max() < 1e-9


def test_fps_improves_smooth_sphere_over_random():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.12)
    random = generate(mesh, _variant="random")
    uniform = generate(mesh, _variant="fps")
    assert (
        uniform.diagnostics["surface_uniformity"]["site_distance_m"]["p95"]
        < random.diagnostics["surface_uniformity"]["site_distance_m"]["p95"]
    )
    assert np.array_equal(uniform.spheres, generate(mesh, _variant="fps").spheres)
    assert len(uniform.spheres) == 64
    assert not uniform.diagnostics["hull_structure"]["box"]["enabled"]


def test_closed_rim_and_feature_budget_without_semantic_hint():
    profile = [[0, 0], [0.05, 0], [0.14, 0.1], [0.13, 0.1], [0.045, 0.008], [0, 0.008]]
    mesh = trimesh.creation.revolve(profile, sections=64)
    result = generate(mesh)
    loops = result.diagnostics["feature_diagnostics"]["rim_loops"]
    assert loops and max(r["site_count_on_feature"] for r in loops) >= 8
    assert result.diagnostics["selection"]["actual_feature_seeds"] <= 28
    assert result.regions.count("generic") >= 36
    assert all(r["arc_gap_m"]["max"] <= r["length_m"] for r in loops)


def test_radius_shared_bounded_count_independent_and_no_legacy_metrics():
    mesh = cube()
    a, b = generate(mesh, {"max_spheres": 32}), generate(mesh)
    assert a.spheres[0, 3] == b.spheres[0, 3]
    mesh.apply_scale(100)
    assert shared_radius(convex_hull(mesh)) == 0.012
    for forbidden in (
        "material_depth",
        "missing_material_components",
        "heldout_manual_labels_only",
        "free_space",
        "max_outward_offset_m",
    ):
        assert forbidden not in b.diagnostics
    json.dumps(b.diagnostics, allow_nan=False)


def test_input_is_not_mutated_and_old_generator_not_called(monkeypatch):
    from asset_collision_spheres.algorithms import auto_geometry, joint_selection

    def fail(*a, **kw):
        raise AssertionError("Old material path called")

    monkeypatch.setattr(auto_geometry, "generate_auto_geometry_spheres", fail)
    monkeypatch.setattr(joint_selection, "generate", fail)
    mesh = cube()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    generate(mesh)
    assert np.array_equal(vertices, mesh.vertices) and np.array_equal(faces, mesh.faces)


def test_independent_obstacle_contacts_report_sparse_misses():
    mesh = cube()
    hull = convex_hull(mesh)
    result = generate(mesh)
    report = obstacle_approaches(hull, result.spheres, analyze(hull))
    assert len(report["cases"]) == 72
    assert report["hull_hits_sphere_misses"] > 0
    # Along +X, finite box centered on face: grid's face-center hole is missed.
    assert any(
        r["hull_hit_but_spheres_miss"] and np.allclose(r["normal"], [1, 0, 0])
        for r in report["cases"]
    )


def test_convex_loader_keeps_old_material_validation(tmp_path):
    from asset_collision_spheres.loaders.mesh import load_mesh

    mesh = cube()
    mesh.update_faces(np.arange(len(mesh.faces) - 2))
    path = tmp_path / "open.obj"
    path.write_text(mesh.export(file_type="obj"))
    with pytest.raises(ValueError):
        load_mesh(path, unit_scale_m=1)
    generate(load_mesh(path, unit_scale_m=1, require_material=False))


def test_rounded_proxy_never_fabricates_sharp_corners():
    profile = [[0, 0], [0.12, 0], [0.12, 0.2], [0.11, 0.2], [0.11, 0.01], [0, 0.01]]
    mesh = trimesh.creation.revolve(profile, sections=64)
    xy = mesh.vertices[:, :2].copy()
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    radius = np.linalg.norm(xy, axis=1)
    mesh.vertices[:, 0] = radius * np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** 0.5
    mesh.vertices[:, 1] = radius * np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** 0.5
    result = generate(mesh)
    assert result.diagnostics["surface_site_max_error_m"] < 1e-8
    if result.diagnostics["sampling_strategy"] == "grid":
        assert result.diagnostics["selection"]["replaced_by_generic_target_ids"]
        assert "corner" not in result.regions
        assert result.diagnostics["review_required"]


def test_real_asset_grid_rotation_and_rim_regression():
    from pathlib import Path
    import importlib.util
    from asset_collision_spheres.paths import workspace_root

    if (
        workspace_root(required=False) is None
        or importlib.util.find_spec("pxr") is None
    ):
        pytest.skip("Optional workspace assets/USD are unavailable")
    script = Path(__file__).resolve().parents[1] / "examples/run_joint_experiments.py"
    spec = importlib.util.spec_from_file_location("convex_v3_test_cases", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        mesh, _ = module.load_case("box")
    except (OSError, ValueError):
        pytest.skip("Optional box asset unavailable")
    result = generate(mesh)
    transform = trimesh.transformations.euler_matrix(0.43, -0.68, 0.9)
    transform[:3, 3] = [0.4, -0.8, 0.3]
    rotated = mesh.copy()
    rotated.apply_transform(transform)
    other = generate(rotated)
    inverse = (other.spheres[:, :3] - transform[:3, 3]) @ transform[:3, :3]
    assert cKDTree(result.spheres[:, :3]).query(inverse)[0].max() < 1e-8
    for name in ("bowl", "mug"):
        mesh, _ = module.load_case(name)
        data = generate(mesh)
        assert data.regions.count("rim") >= 12
        assert data.diagnostics["sampling_strategy"] == "features"
        assert len(data.spheres) == 64
        if name == "bowl":
            structure = data.diagnostics["hull_structure"]
            largest = max(structure["patches"], key=lambda p: p["area_m2"])
            assert largest["area_m2"] / data.diagnostics["hull_area_m2"] > 0.32
            loop = next(
                p for p in structure["rim_loops"] if p["patch_id"] == largest["id"]
            )
            assert (
                loop["sharp_boundary_fraction"] > 0.95
            )  # no chord through cap interior


def test_convex_public_top_level_and_cli_open_mesh(tmp_path):
    from asset_collision_spheres import generate_convex_surface_spheres
    from asset_collision_spheres.cli import main

    assert generate_convex_surface_spheres is generate
    mesh = cube()
    mesh.update_faces(np.arange(len(mesh.faces) - 2))
    path, config, output = (
        tmp_path / "open.obj",
        tmp_path / "config.json",
        tmp_path / "spheres.json",
    )
    path.write_text(mesh.export(file_type="obj"))
    config.write_text('{"mode":"convex_surface","max_spheres":32}')
    assert (
        main(
            [
                "--mesh",
                str(path),
                "--unit-scale-m",
                "1",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    saved = json.loads(output.read_text())
    assert len(saved["spheres_m"]) <= 32
    assert saved["diagnostics"]["target_geometry"] == "convex_hull"


def test_usd_nontriangle_open_geometry_retains_units_transform(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom
    from asset_collision_spheres.loaders.usd import load_material_mesh

    path = tmp_path / "open.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    prim = UsdGeom.Mesh.Define(stage, "/Object")
    prim.CreatePointsAttr(
        [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0), (0, 0, 100)]
    )
    prim.CreateFaceVertexCountsAttr([4, 3])
    prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 0, 1, 4])
    prim.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    stage.GetRootLayer().Save()
    preset = {
        "source": {"usd_path": str(path), "mesh_prim": "/Object", "unit_scale_m": 0.001}
    }
    with pytest.raises(ValueError, match="triangulated"):
        load_material_mesh(preset)
    mesh, _ = load_material_mesh(preset, require_material=False)
    assert np.allclose(mesh.bounds, [[0.01, 0.02, 0.03], [0.11, 0.12, 0.13]])
    assert not mesh.is_watertight
    generate(mesh)
