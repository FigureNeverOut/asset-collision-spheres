from __future__ import annotations

import numpy as np
import pytest
import trimesh

from asset_collision_spheres.mesh_region_spheres import (
    generate_mesh_region_spheres,
    validate_request,
)
from asset_collision_spheres.solid_box_regions import make_solid_box_region_profile


@pytest.mark.parametrize("extents", [(0.23, 0.02, 0.16), (0.01, 0.3, 0.09), (0.12, 0.18, 0.015)])
@pytest.mark.parametrize("budget", [9, 16, 64, 127, 256])
def test_profiles_derive_axes_bounds_and_exact_budget(extents, budget):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation([0.3, -0.5, 0.7])
    before_vertices, before_faces = mesh.vertices.copy(), mesh.faces.copy()
    profile = make_solid_box_region_profile(mesh, budget=budget)
    assert profile.facts["thin_axis_index"] == np.argmin(extents)
    assert sum(profile.facts["region_counts"]) == budget
    assert min(profile.facts["region_counts"]) >= 1
    if budget == 64:
        assert profile.facts["region_counts"] == [4, 8, 4, 8, 16, 8, 4, 8, 4]
    validate_request(mesh, profile.config)
    np.testing.assert_array_equal(mesh.vertices, before_vertices)
    np.testing.assert_array_equal(mesh.faces, before_faces)
    assert not profile.facts["geometry_replaced"]
    assert not profile.facts["production_adapter_enabled"]


@pytest.mark.parametrize("budget", [True, 8, 16.0, 257])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="exact budget"):
        make_solid_box_region_profile(trimesh.creation.box(extents=(0.2, 0.02, 0.1)), budget=budget)


def test_thick_curved_rotated_open_and_concave_meshes_are_not_silently_boxified():
    with pytest.raises(ValueError, match="Thick"):
        make_solid_box_region_profile(trimesh.creation.box())
    disc = trimesh.creation.cylinder(radius=0.1, height=0.01)
    with pytest.raises(ValueError, match="box-like"):
        make_solid_box_region_profile(disc)
    rotated = trimesh.creation.box(extents=(0.2, 0.1, 0.01))
    rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1]))
    with pytest.raises(ValueError, match="box-like"):
        make_solid_box_region_profile(rotated)
    open_mesh = trimesh.creation.box(extents=(0.2, 0.1, 0.01))
    open_mesh.update_faces(np.arange(len(open_mesh.faces) - 1))
    with pytest.raises(ValueError, match="closed"):
        make_solid_box_region_profile(open_mesh)
    parts = [trimesh.creation.box(extents=(0.07, 0.1, 0.01)) for _ in range(2)]
    parts[0].apply_translation([-0.07, 0, 0])
    parts[1].apply_translation([0.07, 0, 0])
    with pytest.raises(ValueError, match="box-like"):
        make_solid_box_region_profile(trimesh.util.concatenate(parts))


@pytest.mark.parametrize("thin_axis", [0, 1, 2])
def test_generated_spheres_keep_mesh_and_fit_expansion_bound(thin_axis):
    extents = np.array([0.12, 0.12, 0.12])
    extents[thin_axis] = 0.015
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation([0.1, -0.2, 0.3])
    config = make_solid_box_region_profile(mesh, budget=16).config
    config["candidate_sample_count"] = 1500
    result = generate_mesh_region_spheres(mesh, config)
    assert result.spheres.shape == (16, 4)
    centers, radii = result.spheres[:, :3], result.spheres[:, 3]
    # Exact box signed distance, independent of the generator's mesh query.
    depths = np.min(extents / 2 - np.abs(centers - mesh.bounds.mean(0)), axis=1)
    assert np.all(depths >= 0)
    assert np.max(radii - depths) <= config["max_outward_offset_m"] + 1e-8
    assert {name: result.regions.count(name) for name in set(result.regions)} == {
        item["name"]: item["sphere_count"] for item in config["regions"]
    }
