from __future__ import annotations

from dataclasses import replace

import pytest

from asset_collision_spheres.attachment_spheres import (
    AttachmentShapeHint,
    AttachmentSphereBudgetError,
    AttachmentSphereConfigurationError,
    AttachmentValidationStatus,
    LocalCollisionSphere,
    generate_structure_aware_spheres,
    parse_structure_aware_request,
    validate_structure_aware_spheres,
)


def _request(**updates: object):
    raw: dict[str, object] = {
        "shape_hint": "solid_box",
        "dimensions_m": [0.2, 0.1, 0.08],
        "sphere_budget": 16,
        "task_profile": "narrow_passage",
        "max_outward_offset_m": 0.001,
    }
    raw.update(updates)
    return parse_structure_aware_request(raw, default_budget=16, default_seed=7)


def test_solid_box_represents_faces_and_corners_with_finite_positive_spheres() -> None:
    request = _request()

    result = generate_structure_aware_spheres(
        request,
        reference_bounds_m=((-0.1, -0.05, -0.04), (0.1, 0.05, 0.04)),
    )

    assert result.shape_hint is AttachmentShapeHint.SOLID_BOX
    assert len(result.spheres) == 16
    assert {sphere.region for sphere in result.spheres} >= {"faces", "corners"}
    assert all(sphere.radius_m > 0.0 for sphere in result.spheres)
    assert result.validation.status is AttachmentValidationStatus.CRITICAL_TESTS_PASSED
    assert result.validation.ordinary_sample_count > result.validation.critical_test_count


def test_solid_box_adapts_corner_representatives_to_small_collision_probe() -> None:
    request = _request(
        dimensions_m=[0.12, 0.08, 0.08],
        validation_probe_radius_m=0.006,
    )

    result = generate_structure_aware_spheres(
        request,
        reference_bounds_m=((-0.06, -0.04, -0.04), (0.06, 0.04, 0.04)),
    )

    corners = tuple(sphere for sphere in result.spheres if sphere.region == "corners")
    assert len(corners) == 8
    assert all(sphere.radius_m == pytest.approx(0.006) for sphere in corners)
    assert result.validation.status is AttachmentValidationStatus.CRITICAL_TESTS_PASSED
    assert result.validation.critical_misses == ()


@pytest.mark.parametrize(
    ("raw", "bounds", "expected_regions"),
    [
        (
            {
                "shape_hint": "open_box",
                "dimensions_m": [0.2, 0.14, 0.1],
                "wall_thickness_m": 0.01,
                "bottom_thickness_m": 0.012,
                "sphere_budget": 16,
                "task_profile": "cavity_access",
                "protect_cavity": True,
                "axis": "z",
                "open_direction": "+",
                "max_outward_offset_m": 0.001,
            },
            ((-0.1, -0.07, -0.05), (0.1, 0.07, 0.05)),
            {"walls", "bottom", "rim"},
        ),
        (
            {
                "shape_hint": "cup",
                "bottom_radius_m": 0.035,
                "top_radius_m": 0.04,
                "height_m": 0.12,
                "wall_thickness_m": 0.004,
                "bottom_thickness_m": 0.006,
                "sphere_budget": 16,
                "task_profile": "cavity_access",
                "protect_cavity": True,
                "axis": "z",
                "max_outward_offset_m": 0.001,
            },
            ((-0.04, -0.04, -0.06), (0.04, 0.04, 0.06)),
            {"wall", "bottom", "rim"},
        ),
    ],
)
def test_hollow_shapes_keep_structural_components_and_protected_cavity(
    raw: dict[str, object],
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    expected_regions: set[str],
) -> None:
    request = parse_structure_aware_request(raw, default_budget=16, default_seed=9)

    result = generate_structure_aware_spheres(request, reference_bounds_m=bounds)

    assert len(result.spheres) <= request.sphere_budget
    assert {sphere.region for sphere in result.spheres} >= expected_regions
    assert result.validation.status is AttachmentValidationStatus.CRITICAL_TESTS_PASSED
    assert result.validation.cavity_intrusion_count == 0


def test_critical_regions_validate_allocated_sites_and_leave_other_gaps_diagnostic() -> None:
    request = parse_structure_aware_request(
        {
            "shape_hint": "cup",
            "radius_m": 0.04,
            "height_m": 0.12,
            "wall_thickness_m": 0.004,
            "bottom_thickness_m": 0.006,
            "sphere_budget": 16,
            "task_profile": "cavity_access",
            "critical_regions": ["rim", "wall", "bottom"],
            "protect_cavity": True,
            "validation_probe_radius_m": 0.004,
            "max_outward_offset_m": 0.001,
        },
        default_budget=16,
        default_seed=0,
    )

    result = generate_structure_aware_spheres(
        request,
        reference_bounds_m=((-0.04, -0.04, -0.06), (0.04, 0.04, 0.06)),
    )

    assert len(result.spheres) == 16
    assert result.validation.status is AttachmentValidationStatus.CRITICAL_TESTS_PASSED
    assert result.validation.critical_misses == ()
    assert result.validation.ordinary_coverage < 1.0


def test_missing_open_box_wall_is_detected_as_reference_collision_false_negative() -> None:
    request = parse_structure_aware_request(
        {
            "shape_hint": "open_box",
            "dimensions_m": [0.2, 0.14, 0.1],
            "wall_thickness_m": 0.01,
            "bottom_thickness_m": 0.012,
            "sphere_budget": 16,
            "protect_cavity": True,
            "validation_probe_radius_m": 0.002,
            "max_outward_offset_m": 0.001,
        },
        default_budget=16,
        default_seed=0,
    )
    bounds = ((-0.1, -0.07, -0.05), (0.1, 0.07, 0.05))
    result = generate_structure_aware_spheres(request, reference_bounds_m=bounds)
    damaged = tuple(
        sphere
        for sphere in result.spheres
        if not (sphere.region == "walls" and sphere.center_m[0] < 0.0)
    )

    validation = validate_structure_aware_spheres(request, damaged, reference_bounds_m=bounds)

    assert validation.status is AttachmentValidationStatus.VALIDATION_FAILED
    assert any(probe.startswith("wall-u-:") for probe in validation.critical_misses)


def test_large_center_sphere_is_detected_as_cup_cavity_false_positive() -> None:
    request = parse_structure_aware_request(
        {
            "shape_hint": "cup",
            "radius_m": 0.04,
            "height_m": 0.12,
            "wall_thickness_m": 0.004,
            "bottom_thickness_m": 0.006,
            "sphere_budget": 16,
            "protect_cavity": True,
            "cavity_tolerance_m": 0.0001,
            "max_outward_offset_m": 0.001,
        },
        default_budget=16,
        default_seed=0,
    )
    bounds = ((-0.04, -0.04, -0.06), (0.04, 0.04, 0.06))
    result = generate_structure_aware_spheres(request, reference_bounds_m=bounds)
    blocked = (*result.spheres, LocalCollisionSphere((0.0, 0.0, 0.02), 0.02, "interior"))

    validation = validate_structure_aware_spheres(request, blocked, reference_bounds_m=bounds)

    assert validation.status is AttachmentValidationStatus.VALIDATION_FAILED
    assert validation.cavity_intrusion_count == 1
    assert validation.cavity_intrusion_max_m == pytest.approx(0.02)


def test_budget_and_shape_declarations_fail_explicitly() -> None:
    with pytest.raises(AttachmentSphereBudgetError, match="minimum_structural_budget=14"):
        generate_structure_aware_spheres(
            _request(sphere_budget=8),
            reference_bounds_m=((-0.1, -0.05, -0.04), (0.1, 0.05, 0.04)),
        )
    with pytest.raises(AttachmentSphereConfigurationError, match="requires wall_thickness"):
        parse_structure_aware_request(
            {"shape_hint": "open_box", "dimensions_m": [0.2, 0.1, 0.08]},
            default_budget=16,
            default_seed=0,
        )
    with pytest.raises(AttachmentSphereConfigurationError, match="unknown structure-aware field"):
        _request(unknown_geometry_guess=True)


def test_scaled_reference_dimensions_and_axis_direction_are_checked() -> None:
    request = parse_structure_aware_request(
        {
            "shape_hint": "open_box",
            "dimensions_m": [0.1, 0.12, 0.2],
            "wall_thickness_m": 0.01,
            "bottom_thickness_m": 0.012,
            "axis": "x",
            "open_direction": "-",
            "sphere_budget": 16,
            "max_outward_offset_m": 0.001,
        },
        default_budget=16,
        default_seed=0,
    )
    result = generate_structure_aware_spheres(
        request,
        reference_bounds_m=((-0.05, -0.06, -0.1), (0.05, 0.06, 0.1)),
    )
    rim = tuple(sphere for sphere in result.spheres if sphere.region == "rim")

    assert rim
    assert all(sphere.center_m[0] < 0.0 for sphere in rim)
    with pytest.raises(AttachmentSphereConfigurationError, match=r"captured=0\.2m, declared=0\.1m"):
        generate_structure_aware_spheres(
            request,
            reference_bounds_m=((-0.1, -0.06, -0.1), (0.1, 0.06, 0.1)),
        )


def test_validation_uses_true_nearest_sphere_surface_gap() -> None:
    request = _request()
    result = generate_structure_aware_spheres(
        request,
        reference_bounds_m=((-0.1, -0.05, -0.04), (0.1, 0.05, 0.04)),
    )
    shifted = tuple(
        replace(sphere, center_m=(sphere.center_m[0] + 0.5, *sphere.center_m[1:]))
        for sphere in result.spheres
    )

    validation = validate_structure_aware_spheres(
        request,
        shifted,
        reference_bounds_m=((-0.1, -0.05, -0.04), (0.1, 0.05, 0.04)),
    )

    assert validation.max_uncovered_gap_m > 0.3
    assert validation.status is AttachmentValidationStatus.VALIDATION_FAILED
