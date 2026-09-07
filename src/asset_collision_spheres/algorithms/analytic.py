"""Deterministic structure-aware collision spheres for attached payloads.

The module is deliberately CPU-only.  It builds spheres in the payload's
local frame and validates them against independent analytic material models;
the cuRobo adapter is responsible only for the later rigid attachment
transform and native tensor update.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

Vector3 = tuple[float, float, float]
Bounds3 = tuple[Vector3, Vector3]

ALGORITHM_VERSION = "structure-aware-attachment-spheres-v2"
_EPSILON = 1.0e-9


class AttachmentShapeHint(StrEnum):
    """Specialized shapes supported by the first implementation."""

    SOLID_BOX = "solid_box"
    OPEN_BOX = "open_box"
    CUP = "cup"
    GENERIC = "generic"


class AttachmentTaskProfile(StrEnum):
    """Small, explicit task vocabulary used only for sphere allocation."""

    GENERAL = "general"
    TRANSPORT = "transport"
    NARROW_PASSAGE = "narrow_passage"
    CAVITY_ACCESS = "cavity_access"


class AttachmentValidationStatus(StrEnum):
    """Truthful validation state reported with every generated set."""

    NOT_VALIDATED = "not_validated"
    CRITICAL_TESTS_PASSED = "critical_tests_passed"
    VALIDATION_FAILED = "validation_failed"


class AttachmentSphereConfigurationError(ValueError):
    """Raised when a dedicated shape declaration is incomplete or invalid."""


class AttachmentSphereBudgetError(AttachmentSphereConfigurationError):
    """Raised when a budget cannot represent every mandatory component."""

    def __init__(self, shape_hint: AttachmentShapeHint, requested: int, required: int) -> None:
        super().__init__(
            f"{shape_hint.value} sphere budget is insufficient: requested={requested}, "
            f"minimum_structural_budget={required}"
        )
        self.shape_hint = shape_hint
        self.requested = requested
        self.required = required


@dataclass(frozen=True, slots=True)
class LocalCollisionSphere:
    """One positive-radius sphere in payload-local metres."""

    center_m: Vector3
    radius_m: float
    region: str

    def __post_init__(self) -> None:
        center = _vector3(self.center_m, "sphere.center_m", positive=False)
        radius = _positive_float(self.radius_m, "sphere.radius_m")
        region = str(self.region).strip()
        if not region:
            raise AttachmentSphereConfigurationError("sphere.region must be non-empty")
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "region", region)


@dataclass(frozen=True, slots=True)
class StructureAwareSphereRequest:
    """Normalized dedicated-shape request expressed entirely in metres."""

    shape_hint: AttachmentShapeHint
    task_profile: AttachmentTaskProfile
    sphere_budget: int
    max_outward_offset_m: float
    protect_cavity: bool
    dimensions_m: Vector3 | None
    wall_thickness_m: float | None
    bottom_thickness_m: float | None
    axis: str
    open_direction: int
    bottom_radius_m: float | None
    top_radius_m: float | None
    critical_regions: tuple[str, ...]
    validation_probe_radius_m: float | None
    cavity_tolerance_m: float
    dimension_tolerance_m: float
    seed: int


@dataclass(frozen=True, slots=True)
class StructureValidation:
    """Independent analytic probe and free-space validation output."""

    status: AttachmentValidationStatus
    critical_test_count: int
    critical_misses: tuple[str, ...]
    ordinary_sample_count: int
    ordinary_coverage: float
    surface_gap_mean_m: float
    surface_gap_p95_m: float
    max_uncovered_gap_m: float
    cavity_intrusion_count: int
    cavity_intrusion_max_m: float
    protrusion_fraction: float
    protrusion_mean_m: float
    protrusion_p95_m: float
    volume_ratio: float


@dataclass(frozen=True, slots=True)
class StructureAwareSphereResult:
    """Generated payload-local spheres plus bounded diagnostics."""

    algorithm_version: str
    shape_hint: AttachmentShapeHint
    task_profile: AttachmentTaskProfile
    requested_budget: int
    spheres: tuple[LocalCollisionSphere, ...]
    generation_time_s: float
    validation: StructureValidation
    support_scope: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    sphere: LocalCollisionSphere
    probe_m: Vector3
    probe_id: str
    mandatory: bool


_COMMON_FIELDS = frozenset(
    {
        "shape_hint",
        "task_profile",
        "sphere_budget",
        "max_outward_offset_m",
        "protect_cavity",
        "dimensions_m",
        "wall_thickness_m",
        "bottom_thickness_m",
        "axis",
        "open_direction",
        "radius_m",
        "bottom_radius_m",
        "top_radius_m",
        "height_m",
        "critical_regions",
        "validation_probe_radius_m",
        "cavity_tolerance_m",
        "dimension_tolerance_m",
        "seed",
        "fallback",
        "has_extra_structure",
    }
)

_REGIONS = {
    AttachmentShapeHint.SOLID_BOX: frozenset({"faces", "edges", "corners", "bottom", "exterior"}),
    AttachmentShapeHint.OPEN_BOX: frozenset({"walls", "bottom", "rim", "seams", "interior", "exterior"}),
    AttachmentShapeHint.CUP: frozenset({"wall", "bottom", "rim", "interior", "exterior"}),
    AttachmentShapeHint.GENERIC: frozenset(),
}


def structure_config_fallback(raw: Mapping[str, object]) -> str | None:
    """Return a validated per-object fallback override when one is present."""

    value = raw.get("fallback")
    if value is None:
        return None
    if type(value) is not str or value not in {"baseline", "error"}:
        raise AttachmentSphereConfigurationError("fallback must be 'baseline' or 'error'")
    return value


def parse_structure_aware_request(
    raw: Mapping[str, object],
    *,
    default_budget: int,
    default_seed: int,
) -> StructureAwareSphereRequest:
    """Strictly decode one per-object dedicated-shape configuration."""

    unknown = tuple(sorted(set(raw) - _COMMON_FIELDS))
    if unknown:
        raise AttachmentSphereConfigurationError(f"unknown structure-aware field {unknown[0]!r}")
    raw_shape_hint = raw.get("shape_hint", "generic")
    try:
        if type(raw_shape_hint) is not str:
            raise TypeError
        shape_hint = AttachmentShapeHint(raw_shape_hint)
    except (TypeError, ValueError) as error:
        raise AttachmentSphereConfigurationError(
            "shape_hint must be solid_box, open_box, cup, or generic"
        ) from error
    raw_task_profile = raw.get("task_profile", "general")
    try:
        if type(raw_task_profile) is not str:
            raise TypeError
        task_profile = AttachmentTaskProfile(raw_task_profile)
    except (TypeError, ValueError) as error:
        raise AttachmentSphereConfigurationError(
            "task_profile must be general, transport, narrow_passage, or cavity_access"
        ) from error
    budget_value = raw.get("sphere_budget", default_budget)
    if type(budget_value) is not int or budget_value < 1:
        raise AttachmentSphereConfigurationError("sphere_budget must be a positive exact integer")
    if budget_value > default_budget:
        raise AttachmentSphereConfigurationError(
            f"sphere_budget={budget_value} exceeds native attached-object capacity={default_budget}"
        )
    max_outward = _nonnegative_float(raw.get("max_outward_offset_m", 0.0), "max_outward_offset_m")
    protect_cavity = raw.get("protect_cavity", shape_hint in {AttachmentShapeHint.OPEN_BOX, AttachmentShapeHint.CUP})
    if type(protect_cavity) is not bool:
        raise AttachmentSphereConfigurationError("protect_cavity must be a bool")
    axis = raw.get("axis", "z")
    if type(axis) is not str or axis not in {"x", "y", "z"}:
        raise AttachmentSphereConfigurationError("axis must be 'x', 'y', or 'z'")
    raw_direction = raw.get("open_direction", "+")
    if raw_direction not in {"+", "-", 1, -1} or type(raw_direction) is bool:
        raise AttachmentSphereConfigurationError("open_direction must be '+', '-', 1, or -1")
    open_direction = 1 if raw_direction in {"+", 1} else -1
    dimensions = raw.get("dimensions_m")
    dimensions_m = None if dimensions is None else _vector3(dimensions, "dimensions_m", positive=True)
    wall = raw.get("wall_thickness_m")
    wall_thickness = None if wall is None else _positive_float(wall, "wall_thickness_m")
    bottom = raw.get("bottom_thickness_m", wall_thickness)
    bottom_thickness = None if bottom is None else _positive_float(bottom, "bottom_thickness_m")
    radius = raw.get("radius_m")
    bottom_radius = raw.get("bottom_radius_m", radius)
    top_radius = raw.get("top_radius_m", radius)
    bottom_radius_m = None if bottom_radius is None else _positive_float(bottom_radius, "bottom_radius_m")
    top_radius_m = None if top_radius is None else _positive_float(top_radius, "top_radius_m")
    height = raw.get("height_m")
    if shape_hint is AttachmentShapeHint.CUP:
        if height is None:
            raise AttachmentSphereConfigurationError("cup requires height_m")
        height_m = _positive_float(height, "height_m")
        dimensions_m = (2.0 * max(bottom_radius_m or 0.0, top_radius_m or 0.0),) * 2 + (height_m,)
        if bottom_radius_m is None or top_radius_m is None:
            raise AttachmentSphereConfigurationError(
                "cup requires radius_m or both bottom_radius_m and top_radius_m"
            )
        if wall_thickness is None or bottom_thickness is None:
            raise AttachmentSphereConfigurationError("cup requires wall_thickness_m and bottom_thickness_m")
        if wall_thickness >= min(bottom_radius_m, top_radius_m):
            raise AttachmentSphereConfigurationError("cup wall thickness must be smaller than both outer radii")
        if bottom_thickness >= height_m:
            raise AttachmentSphereConfigurationError("cup bottom thickness must be smaller than height")
    elif shape_hint in {AttachmentShapeHint.SOLID_BOX, AttachmentShapeHint.OPEN_BOX}:
        if dimensions_m is None:
            raise AttachmentSphereConfigurationError(f"{shape_hint.value} requires dimensions_m")
        if shape_hint is AttachmentShapeHint.OPEN_BOX:
            if wall_thickness is None or bottom_thickness is None:
                raise AttachmentSphereConfigurationError(
                    "open_box requires wall_thickness_m and bottom_thickness_m"
                )
            axial = {"x": 0, "y": 1, "z": 2}[axis]
            radial = tuple(index for index in range(3) if index != axial)
            if 2.0 * wall_thickness >= min(dimensions_m[index] for index in radial):
                raise AttachmentSphereConfigurationError("open_box walls leave no interior width")
            if bottom_thickness >= dimensions_m[axial]:
                raise AttachmentSphereConfigurationError("open_box bottom leaves no interior depth")
    critical_value = raw.get("critical_regions", ())
    if not isinstance(critical_value, (list, tuple)) or any(
        type(item) is not str or not item for item in critical_value
    ):
        raise AttachmentSphereConfigurationError("critical_regions must be a list of non-empty strings")
    critical_regions = tuple(cast(Sequence[str], critical_value))
    invalid_regions = tuple(sorted(set(critical_regions) - _REGIONS[shape_hint]))
    if invalid_regions:
        raise AttachmentSphereConfigurationError(
            f"critical region {invalid_regions[0]!r} is unsupported for {shape_hint.value}"
        )
    probe = raw.get("validation_probe_radius_m")
    probe_radius = None if probe is None else _positive_float(probe, "validation_probe_radius_m")
    cavity_tolerance = _nonnegative_float(raw.get("cavity_tolerance_m", 1.0e-6), "cavity_tolerance_m")
    dimension_tolerance = _nonnegative_float(
        raw.get("dimension_tolerance_m", 1.0e-4),
        "dimension_tolerance_m",
    )
    seed = raw.get("seed", default_seed)
    if type(seed) is not int:
        raise AttachmentSphereConfigurationError("seed must be an exact integer")
    return StructureAwareSphereRequest(
        shape_hint=shape_hint,
        task_profile=task_profile,
        sphere_budget=budget_value,
        max_outward_offset_m=max_outward,
        protect_cavity=protect_cavity,
        dimensions_m=dimensions_m,
        wall_thickness_m=wall_thickness,
        bottom_thickness_m=bottom_thickness,
        axis=axis,
        open_direction=open_direction,
        bottom_radius_m=bottom_radius_m,
        top_radius_m=top_radius_m,
        critical_regions=critical_regions,
        validation_probe_radius_m=probe_radius,
        cavity_tolerance_m=cavity_tolerance,
        dimension_tolerance_m=dimension_tolerance,
        seed=seed,
    )


def generate_structure_aware_spheres(
    request: StructureAwareSphereRequest,
    *,
    reference_bounds_m: Bounds3,
) -> StructureAwareSphereResult:
    """Generate and independently validate one supported analytic structure."""

    if request.shape_hint is AttachmentShapeHint.GENERIC:
        raise AttachmentSphereConfigurationError("generic shape_hint must use the baseline generator")
    started = time.perf_counter()
    bounds = _bounds(reference_bounds_m)
    _validate_reference_bounds(request, bounds)
    if request.shape_hint is AttachmentShapeHint.SOLID_BOX:
        candidates, support_scope = _solid_box_candidates(request, bounds)
    elif request.shape_hint is AttachmentShapeHint.OPEN_BOX:
        candidates, support_scope = _open_box_candidates(request, bounds)
    else:
        candidates, support_scope = _cup_candidates(request, bounds)
    mandatory_count = sum(candidate.mandatory for candidate in candidates)
    if request.sphere_budget < mandatory_count:
        raise AttachmentSphereBudgetError(request.shape_hint, request.sphere_budget, mandatory_count)
    ordered = _ordered_candidates(candidates, request)
    selected = tuple(candidate.sphere for candidate in ordered[: request.sphere_budget])
    validation = validate_structure_aware_spheres(request, selected, reference_bounds_m=bounds)
    return StructureAwareSphereResult(
        algorithm_version=ALGORITHM_VERSION,
        shape_hint=request.shape_hint,
        task_profile=request.task_profile,
        requested_budget=request.sphere_budget,
        spheres=selected,
        generation_time_s=time.perf_counter() - started,
        validation=validation,
        support_scope=support_scope,
    )


def validate_structure_aware_spheres(
    request: StructureAwareSphereRequest,
    spheres: Sequence[LocalCollisionSphere],
    *,
    reference_bounds_m: Bounds3,
) -> StructureValidation:
    """Compare spheres with separate analytic collision and cavity probes."""

    values = tuple(spheres)
    if not values or any(not isinstance(item, LocalCollisionSphere) for item in values):
        raise AttachmentSphereConfigurationError("validation requires positive LocalCollisionSphere values")
    bounds = _bounds(reference_bounds_m)
    _validate_reference_bounds(request, bounds)
    distance_to_material: Callable[[Vector3], float]
    cavity_distance: Callable[[Vector3], float] | None
    if request.shape_hint is AttachmentShapeHint.SOLID_BOX:
        candidates, _ = _solid_box_candidates(request, bounds)

        def solid_distance_to_material(point: Vector3) -> float:
            return _distance_to_box(point, bounds)

        distance_to_material = solid_distance_to_material
        material_volume = math.prod(cast(Vector3, request.dimensions_m))
        cavity_distance = None
    elif request.shape_hint is AttachmentShapeHint.OPEN_BOX:
        candidates, _ = _open_box_candidates(request, bounds)
        components, cavity = _open_box_material_and_cavity(request, bounds)

        def open_box_distance_to_material(point: Vector3) -> float:
            return min(_distance_to_box(point, component) for component in components)

        def open_box_cavity_distance(point: Vector3) -> float:
            return _distance_to_box(point, cavity)

        distance_to_material = open_box_distance_to_material
        material_volume = sum(_box_volume(component) for component in components)
        cavity_distance = open_box_cavity_distance
    elif request.shape_hint is AttachmentShapeHint.CUP:
        candidates, _ = _cup_candidates(request, bounds)
        material_polygon, cavity_polygon = _cup_polygons(request)
        center = _bounds_center(bounds)
        axis_index, radial = _axis_layout(request.axis)

        def cup_coordinates(point: Vector3) -> tuple[float, float]:
            rho = math.hypot(point[radial[0]] - center[radial[0]], point[radial[1]] - center[radial[1]])
            z = request.open_direction * (point[axis_index] - center[axis_index])
            return rho, z

        def cup_distance_to_material(point: Vector3) -> float:
            return _distance_to_polygon(cup_coordinates(point), material_polygon)

        def cup_cavity_distance(point: Vector3) -> float:
            return _distance_to_polygon(cup_coordinates(point), cavity_polygon)

        distance_to_material = cup_distance_to_material
        cavity_distance = cup_cavity_distance
        material_volume = _cup_material_volume(request)
    else:
        raise AttachmentSphereConfigurationError("generic shape_hint has no analytic reference material")

    probe_radius = request.validation_probe_radius_m
    if probe_radius is None:
        assert request.dimensions_m is not None
        probe_radius = max(0.002, min(request.dimensions_m) * 0.15)
    allocated_probe_ids = frozenset(
        candidate.probe_id
        for candidate in _ordered_candidates(candidates, request)[: request.sphere_budget]
    )
    critical_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.mandatory
        or (
            candidate.probe_id in allocated_probe_ids
            and candidate.sphere.region in request.critical_regions
        )
    )
    critical_probes = _independent_collision_probes(
        critical_candidates,
        probe_radius,
        distance_to_material,
    )
    misses = tuple(
        probe_id
        for probe_id, probe_m in critical_probes
        if not _probe_collides(probe_m, probe_radius, values)
    )
    ordinary_probes = _independent_collision_probes(candidates, probe_radius, distance_to_material)
    ordinary_gaps = tuple(
        max(0.0, min(_distance(probe_m, sphere.center_m) - sphere.radius_m for sphere in values))
        for _, probe_m in ordinary_probes
    )
    covered = sum(gap <= probe_radius + _EPSILON for gap in ordinary_gaps)
    cavity_intrusions: tuple[float, ...] = ()
    if request.protect_cavity and cavity_distance is not None:
        cavity_intrusions = tuple(max(0.0, sphere.radius_m - cavity_distance(sphere.center_m)) for sphere in values)
    failing_intrusions = tuple(value for value in cavity_intrusions if value > request.cavity_tolerance_m + _EPSILON)
    protrusions: list[float] = []
    for sphere in values:
        for direction in _UNIT_DIRECTIONS:
            surface = tuple(
                sphere.center_m[index] + sphere.radius_m * direction[index]
                for index in range(3)
            )
            protrusions.append(distance_to_material(cast(Vector3, surface)))
    failing_protrusions = tuple(
        value for value in protrusions if value > request.max_outward_offset_m + 2.0e-6
    )
    status = (
        AttachmentValidationStatus.CRITICAL_TESTS_PASSED
        if not misses and not failing_intrusions and not failing_protrusions
        else AttachmentValidationStatus.VALIDATION_FAILED
    )
    sphere_volume = sum((4.0 / 3.0) * math.pi * sphere.radius_m**3 for sphere in values)
    return StructureValidation(
        status=status,
        critical_test_count=len(critical_probes),
        critical_misses=misses,
        ordinary_sample_count=len(ordinary_gaps),
        ordinary_coverage=covered / len(ordinary_gaps),
        surface_gap_mean_m=sum(ordinary_gaps) / len(ordinary_gaps),
        surface_gap_p95_m=_percentile(ordinary_gaps, 0.95),
        max_uncovered_gap_m=max(ordinary_gaps, default=0.0),
        cavity_intrusion_count=len(failing_intrusions),
        cavity_intrusion_max_m=max(cavity_intrusions, default=0.0),
        protrusion_fraction=(sum(value > _EPSILON for value in protrusions) / len(protrusions)),
        protrusion_mean_m=sum(protrusions) / len(protrusions),
        protrusion_p95_m=_percentile(tuple(protrusions), 0.95),
        volume_ratio=sphere_volume / material_volume,
    )


def _solid_box_candidates(
    request: StructureAwareSphereRequest,
    bounds: Bounds3,
) -> tuple[tuple[_Candidate, ...], str]:
    dimensions = cast(Vector3, request.dimensions_m)
    half = tuple(value * 0.5 for value in dimensions)
    center = _bounds_center(bounds)
    face_radius = min(dimensions) * 0.25
    detail_radius = min(dimensions) / 6.0
    # A smaller configured reference probe asks the representation to retain
    # correspondingly finer corner contacts.  Shrinking only the corner
    # representatives moves their tangent points closer to the true vertex;
    # keeping the larger edge/face spheres preserves the coarse body envelope.
    corner_radius = min(detail_radius, request.validation_probe_radius_m or detail_radius)
    candidates: list[_Candidate] = []
    for axis in range(3):
        for sign in (-1, 1):
            local = [0.0, 0.0, 0.0]
            probe = [0.0, 0.0, 0.0]
            local[axis] = sign * (half[axis] - face_radius)
            probe[axis] = sign * half[axis]
            region = "bottom" if axis == 2 and sign < 0 else "faces"
            candidates.append(
                _candidate(center, tuple(local), tuple(probe), face_radius, region, f"face-{axis}-{sign}", True)
            )
    corner_signs = tuple(
        (x_sign, y_sign, z_sign)
        for x_sign in (-1, 1)
        for y_sign in (-1, 1)
        for z_sign in (-1, 1)
    )
    for index, corner_sign in enumerate(corner_signs):
        corner_local = tuple(corner_sign[axis] * (half[axis] - corner_radius) for axis in range(3))
        corner_probe = tuple(corner_sign[axis] * half[axis] for axis in range(3))
        candidates.append(
            _candidate(
                center,
                corner_local,
                corner_probe,
                corner_radius,
                "corners",
                f"corner-{index}",
                True,
            )
        )
    for free_axis in range(3):
        fixed_axes = tuple(axis for axis in range(3) if axis != free_axis)
        for first_sign in (-1, 1):
            for second_sign in (-1, 1):
                edge_signs = (first_sign, second_sign)
                local = [0.0, 0.0, 0.0]
                probe = [0.0, 0.0, 0.0]
                for fixed_axis, sign in zip(fixed_axes, edge_signs, strict=True):
                    local[fixed_axis] = sign * (half[fixed_axis] - detail_radius)
                    probe[fixed_axis] = sign * half[fixed_axis]
                candidates.append(
                    _candidate(
                        center,
                        tuple(local),
                        tuple(probe),
                        detail_radius,
                        "edges",
                        f"edge-{free_axis}-{first_sign}-{second_sign}",
                        False,
                    )
                )
    for axis in range(3):
        radial = tuple(index for index in range(3) if index != axis)
        for sign in (-1, 1):
            for u_sign in (-1, 1):
                for v_sign in (-1, 1):
                    local = [0.0, 0.0, 0.0]
                    probe = [0.0, 0.0, 0.0]
                    local[axis] = sign * (half[axis] - face_radius)
                    probe[axis] = sign * half[axis]
                    local[radial[0]] = probe[radial[0]] = u_sign * half[radial[0]] * 0.5
                    local[radial[1]] = probe[radial[1]] = v_sign * half[radial[1]] * 0.5
                    candidates.append(
                        _candidate(
                            center,
                            tuple(local),
                            tuple(probe),
                            face_radius,
                            "faces",
                            f"face-grid-{axis}-{sign}-{u_sign}-{v_sign}",
                            False,
                        )
                    )
    return tuple(candidates), "axis-aligned solid rectangular boxes without omitted branches"


def _open_box_candidates(
    request: StructureAwareSphereRequest,
    bounds: Bounds3,
) -> tuple[tuple[_Candidate, ...], str]:
    dimensions = cast(Vector3, request.dimensions_m)
    wall = cast(float, request.wall_thickness_m)
    bottom_thickness = cast(float, request.bottom_thickness_m)
    axis_index, radial = _axis_layout(request.axis)
    du, dv, dw = dimensions[radial[0]], dimensions[radial[1]], dimensions[axis_index]
    hu, hv, hw = du * 0.5, dv * 0.5, dw * 0.5
    center = _bounds_center(bounds)
    bottom_z = -hw
    bottom_top = bottom_z + bottom_thickness
    wall_radius = wall * 0.5
    bottom_radius = bottom_thickness * 0.5
    candidates: list[_Candidate] = []
    wall_specs = (
        (0, -1, -hu + wall_radius, "u-"),
        (0, 1, hu - wall_radius, "u+"),
        (1, -1, -hv + wall_radius, "v-"),
        (1, 1, hv - wall_radius, "v+"),
    )
    middle_z = (bottom_top + hw) * 0.5
    for wall_axis, sign, wall_center, label in wall_specs:
        local = [0.0, 0.0, middle_z]
        probe = [0.0, 0.0, middle_z]
        local[wall_axis] = wall_center
        probe[wall_axis] = sign * (hu if wall_axis == 0 else hv)
        candidates.append(
            _canonical_candidate(
                request,
                center,
                cast(Vector3, tuple(local)),
                cast(Vector3, tuple(probe)),
                wall_radius,
                "walls",
                f"wall-{label}",
                True,
            )
        )
        rim_local = list(local)
        rim_probe = list(probe)
        rim_local[2] = hw - wall_radius
        rim_probe[2] = hw
        candidates.append(
            _canonical_candidate(
                request,
                center,
                cast(Vector3, tuple(rim_local)),
                cast(Vector3, tuple(rim_probe)),
                wall_radius,
                "rim",
                f"rim-{label}",
                True,
            )
        )
    candidates.append(
        _canonical_candidate(
            request,
            center,
            (0.0, 0.0, bottom_z + bottom_radius),
            (0.0, 0.0, bottom_z),
            bottom_radius,
            "bottom",
            "bottom-center",
            True,
        )
    )
    for u_fraction in (-0.55, 0.55):
        for v_fraction in (-0.55, 0.55):
            candidates.append(
                _canonical_candidate(
                    request,
                    center,
                    (u_fraction * (hu - wall), v_fraction * (hv - wall), bottom_z + bottom_radius),
                    (u_fraction * (hu - wall), v_fraction * (hv - wall), bottom_z),
                    bottom_radius,
                    "bottom",
                    f"bottom-grid-{u_fraction}-{v_fraction}",
                    False,
                )
            )
    for height_fraction in (0.25, 0.75):
        z = bottom_top + height_fraction * (hw - bottom_top)
        for wall_axis, sign, wall_center, label in wall_specs:
            other_axis = 1 - wall_axis
            for other_sign in (-1, 1):
                local = [0.0, 0.0, z]
                probe = [0.0, 0.0, z]
                local[wall_axis] = wall_center
                probe[wall_axis] = sign * (hu if wall_axis == 0 else hv)
                extent = hv if other_axis == 1 else hu
                local[other_axis] = probe[other_axis] = other_sign * max(0.0, extent - wall)
                candidates.append(
                    _canonical_candidate(
                        request,
                        center,
                        cast(Vector3, tuple(local)),
                        cast(Vector3, tuple(probe)),
                        wall_radius,
                        "seams",
                        f"seam-{label}-{height_fraction}-{other_sign}",
                        False,
                    )
                )
    return tuple(candidates), "five-plate rectangular boxes with one configured axial opening and no extra branches"


def _cup_candidates(
    request: StructureAwareSphereRequest,
    bounds: Bounds3,
) -> tuple[tuple[_Candidate, ...], str]:
    height = cast(Vector3, request.dimensions_m)[2]
    bottom_radius = cast(float, request.bottom_radius_m)
    top_radius = cast(float, request.top_radius_m)
    wall = cast(float, request.wall_thickness_m)
    bottom_thickness = cast(float, request.bottom_thickness_m)
    center = _bounds_center(bounds)
    bottom_z = -height * 0.5
    top_z = height * 0.5
    slope = (top_radius - bottom_radius) / height
    sphere_radius = wall / (2.0 * math.sqrt(1.0 + slope * slope))
    bottom_sphere_radius = bottom_thickness * 0.5
    candidates: list[_Candidate] = []

    def outer_radius(z: float) -> float:
        return bottom_radius + (z - bottom_z) * slope

    def add_ring(
        *,
        z: float,
        count: int,
        phase: float,
        region: str,
        mandatory: bool,
        label: str,
    ) -> None:
        outer = outer_radius(z)
        local_radius = outer - wall * 0.5
        for index in range(count):
            angle = phase + 2.0 * math.pi * index / count
            cosine, sine = math.cos(angle), math.sin(angle)
            local = (local_radius * cosine, local_radius * sine, z)
            probe = (outer * cosine, outer * sine, top_z if region == "rim" else z)
            candidates.append(
                _canonical_candidate(
                    request,
                    center,
                    local,
                    probe,
                    sphere_radius,
                    region,
                    f"{label}-{index}",
                    mandatory,
                )
            )

    wall_bottom = bottom_z + bottom_thickness
    add_ring(
        z=(wall_bottom + top_z) * 0.5,
        count=4,
        phase=0.0,
        region="wall",
        mandatory=True,
        label="wall-mid",
    )
    add_ring(
        z=top_z - sphere_radius,
        count=4,
        phase=math.pi * 0.25,
        region="rim",
        mandatory=True,
        label="rim",
    )
    candidates.append(
        _canonical_candidate(
            request,
            center,
            (0.0, 0.0, bottom_z + bottom_sphere_radius),
            (0.0, 0.0, bottom_z),
            bottom_sphere_radius,
            "bottom",
            "bottom-center",
            True,
        )
    )
    add_ring(
        z=bottom_z + bottom_sphere_radius,
        count=4,
        phase=math.pi * 0.25,
        region="bottom",
        mandatory=False,
        label="bottom-ring",
    )
    add_ring(
        z=wall_bottom + 0.2 * (top_z - wall_bottom),
        count=4,
        phase=math.pi * 0.25,
        region="wall",
        mandatory=False,
        label="wall-low",
    )
    add_ring(
        z=wall_bottom + 0.8 * (top_z - wall_bottom),
        count=4,
        phase=0.0,
        region="wall",
        mandatory=False,
        label="wall-high",
    )
    add_ring(
        z=top_z - sphere_radius,
        count=4,
        phase=0.0,
        region="rim",
        mandatory=False,
        label="rim-extra",
    )
    return tuple(candidates), "handle-free circular cylindrical or simple coaxial frustum cups"


def _ordered_candidates(
    candidates: tuple[_Candidate, ...],
    request: StructureAwareSphereRequest,
) -> tuple[_Candidate, ...]:
    mandatory = [candidate for candidate in candidates if candidate.mandatory]
    optional = [candidate for candidate in candidates if not candidate.mandatory]
    preferred = {
        AttachmentTaskProfile.GENERAL: (),
        AttachmentTaskProfile.TRANSPORT: ("bottom", "exterior", "corners", "edges", "walls", "wall"),
        AttachmentTaskProfile.NARROW_PASSAGE: ("corners", "edges", "rim", "walls", "wall", "faces"),
        AttachmentTaskProfile.CAVITY_ACCESS: ("rim", "interior", "wall", "walls", "bottom", "seams"),
    }[request.task_profile]
    priority = tuple(dict.fromkeys((*request.critical_regions, *preferred)))
    order = {region: index for index, region in enumerate(priority)}
    optional.sort(key=lambda candidate: (order.get(candidate.sphere.region, len(order)), candidate.probe_id))
    return tuple((*mandatory, *optional))


def _open_box_material_and_cavity(
    request: StructureAwareSphereRequest,
    bounds: Bounds3,
) -> tuple[tuple[Bounds3, ...], Bounds3]:
    dimensions = cast(Vector3, request.dimensions_m)
    wall = cast(float, request.wall_thickness_m)
    bottom = cast(float, request.bottom_thickness_m)
    axis_index, radial = _axis_layout(request.axis)
    center = _bounds_center(bounds)
    half = tuple(value * 0.5 for value in dimensions)
    low_u, high_u = -half[radial[0]], half[radial[0]]
    low_v, high_v = -half[radial[1]], half[radial[1]]
    low_z, high_z = -half[axis_index], half[axis_index]
    canonical_components = (
        ((low_u, low_v, low_z), (high_u, high_v, low_z + bottom)),
        ((low_u, low_v, low_z + bottom), (low_u + wall, high_v, high_z)),
        ((high_u - wall, low_v, low_z + bottom), (high_u, high_v, high_z)),
        ((low_u + wall, low_v, low_z + bottom), (high_u - wall, low_v + wall, high_z)),
        ((low_u + wall, high_v - wall, low_z + bottom), (high_u - wall, high_v, high_z)),
    )
    components = tuple(_canonical_bounds(request, center, component) for component in canonical_components)
    cavity = _canonical_bounds(
        request,
        center,
        (
            (low_u + wall, low_v + wall, low_z + bottom),
            (high_u - wall, high_v - wall, high_z),
        ),
    )
    return components, cavity


def _cup_polygons(
    request: StructureAwareSphereRequest,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    height = cast(Vector3, request.dimensions_m)[2]
    bottom_radius = cast(float, request.bottom_radius_m)
    top_radius = cast(float, request.top_radius_m)
    wall = cast(float, request.wall_thickness_m)
    bottom = cast(float, request.bottom_thickness_m)
    low, high = -height * 0.5, height * 0.5
    material = (
        (0.0, low),
        (bottom_radius, low),
        (top_radius, high),
        (top_radius - wall, high),
        (bottom_radius - wall, low + bottom),
        (0.0, low + bottom),
    )
    cavity = (
        (0.0, low + bottom),
        (bottom_radius - wall, low + bottom),
        (top_radius - wall, high),
        (0.0, high),
    )
    return material, cavity


def _cup_material_volume(request: StructureAwareSphereRequest) -> float:
    height = cast(Vector3, request.dimensions_m)[2]
    bottom_radius = cast(float, request.bottom_radius_m)
    top_radius = cast(float, request.top_radius_m)
    wall = cast(float, request.wall_thickness_m)
    bottom = cast(float, request.bottom_thickness_m)

    def frustum_volume(length: float, low_radius: float, high_radius: float) -> float:
        return math.pi * length * (low_radius**2 + low_radius * high_radius + high_radius**2) / 3.0

    outer = frustum_volume(height, bottom_radius, top_radius)
    inner_height = height - bottom
    inner_low = bottom_radius + (top_radius - bottom_radius) * bottom / height - wall
    inner = frustum_volume(inner_height, inner_low, top_radius - wall)
    return outer - inner


def _validate_reference_bounds(request: StructureAwareSphereRequest, bounds: Bounds3) -> None:
    actual = tuple(bounds[1][index] - bounds[0][index] for index in range(3))
    assert request.dimensions_m is not None
    if request.shape_hint is AttachmentShapeHint.CUP:
        axis_index, radial = _axis_layout(request.axis)
        expected_radius = max(cast(float, request.bottom_radius_m), cast(float, request.top_radius_m))
        expected = [0.0, 0.0, 0.0]
        expected[axis_index] = request.dimensions_m[2]
        expected[radial[0]] = expected[radial[1]] = 2.0 * expected_radius
    else:
        expected = list(request.dimensions_m)
    mismatch = tuple(
        (index, actual[index], expected[index])
        for index in range(3)
        if abs(actual[index] - expected[index]) > request.dimension_tolerance_m
    )
    if mismatch:
        index, observed, declared = mismatch[0]
        raise AttachmentSphereConfigurationError(
            "declared dimensions do not match the scaled captured collision geometry: "
            f"axis={index}, captured={observed:.9g}m, declared={declared:.9g}m, "
            f"tolerance={request.dimension_tolerance_m:.9g}m"
        )


def _canonical_candidate(
    request: StructureAwareSphereRequest,
    center: Vector3,
    local: tuple[float, float, float],
    probe: tuple[float, float, float],
    radius: float,
    region: str,
    probe_id: str,
    mandatory: bool,
) -> _Candidate:
    return _Candidate(
        LocalCollisionSphere(_from_canonical(request, center, local), radius, region),
        _from_canonical(request, center, probe),
        probe_id,
        mandatory,
    )


def _candidate(
    center: Vector3,
    local: tuple[float, ...],
    probe: tuple[float, ...],
    radius: float,
    region: str,
    probe_id: str,
    mandatory: bool,
) -> _Candidate:
    sphere_center = cast(Vector3, tuple(center[index] + local[index] for index in range(3)))
    probe_center = cast(Vector3, tuple(center[index] + probe[index] for index in range(3)))
    return _Candidate(LocalCollisionSphere(sphere_center, radius, region), probe_center, probe_id, mandatory)


def _from_canonical(
    request: StructureAwareSphereRequest,
    center: Vector3,
    value: tuple[float, float, float],
) -> Vector3:
    axis_index, radial = _axis_layout(request.axis)
    result = [0.0, 0.0, 0.0]
    result[radial[0]] = value[0]
    result[radial[1]] = value[1]
    result[axis_index] = request.open_direction * value[2]
    return cast(Vector3, tuple(center[index] + result[index] for index in range(3)))


def _canonical_bounds(
    request: StructureAwareSphereRequest,
    center: Vector3,
    value: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> Bounds3:
    points = (_from_canonical(request, center, value[0]), _from_canonical(request, center, value[1]))
    minimum = cast(Vector3, tuple(min(point[index] for point in points) for index in range(3)))
    maximum = cast(Vector3, tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def _axis_layout(axis: str) -> tuple[int, tuple[int, int]]:
    axial = {"x": 0, "y": 1, "z": 2}[axis]
    radial = tuple(index for index in range(3) if index != axial)
    return axial, cast(tuple[int, int], radial)


def _probe_collides(center: Vector3, radius: float, spheres: tuple[LocalCollisionSphere, ...]) -> bool:
    return any(_distance(center, sphere.center_m) <= radius + sphere.radius_m + _EPSILON for sphere in spheres)


def _independent_collision_probes(
    candidates: Sequence[_Candidate],
    probe_radius: float,
    distance_to_material: Callable[[Vector3], float],
) -> tuple[tuple[str, Vector3], ...]:
    """Build collision checks distinct from the sphere-allocation samples.

    Each perturbed probe remains a reference-geometry collision because its
    centre is less than one probe radius from a known material point.  The
    analytic distance predicate is still checked so future candidate changes
    cannot weaken that invariant accidentally.
    """

    offsets = (
        (0.08, -0.05, 0.03),
        (-0.06, 0.08, -0.04),
        (0.04, 0.05, -0.08),
    )
    probes: list[tuple[str, Vector3]] = []
    for candidate in candidates:
        for index, coefficients in enumerate(offsets):
            point = cast(
                Vector3,
                tuple(
                    candidate.probe_m[axis] + probe_radius * coefficients[axis]
                    for axis in range(3)
                ),
            )
            if float(distance_to_material(point)) > probe_radius + _EPSILON:
                raise RuntimeError("independent collision probe does not collide with reference material")
            probes.append((f"{candidate.probe_id}:probe-{index}", point))
    return tuple(probes)


def _distance(left: Vector3, right: Vector3) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _distance_to_box(point: Vector3, bounds: Bounds3) -> float:
    outside = tuple(
        max(bounds[0][index] - point[index], 0.0, point[index] - bounds[1][index])
        for index in range(3)
    )
    return math.sqrt(sum(value * value for value in outside))


def _distance_to_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> float:
    if _point_in_polygon(point, polygon):
        return 0.0
    return min(
        _distance_to_segment_2d(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if ((current[1] > y) != (previous[1] > y)) and (
            x < (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1]) + current[0]
        ):
            inside = not inside
        previous = current
    return inside


def _distance_to_segment_2d(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= _EPSILON:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    numerator = (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    fraction = max(0.0, min(1.0, numerator / denominator))
    closest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def _box_volume(bounds: Bounds3) -> float:
    return math.prod(bounds[1][index] - bounds[0][index] for index in range(3))


def _bounds(value: Bounds3) -> Bounds3:
    if not isinstance(value, tuple) or len(value) != 2:
        raise AttachmentSphereConfigurationError("reference_bounds_m must be a (minimum, maximum) tuple")
    minimum = _vector3(value[0], "reference_bounds_m.minimum", positive=False)
    maximum = _vector3(value[1], "reference_bounds_m.maximum", positive=False)
    if any(maximum[index] <= minimum[index] for index in range(3)):
        raise AttachmentSphereConfigurationError("reference bounds must have positive finite extents")
    return minimum, maximum


def _bounds_center(bounds: Bounds3) -> Vector3:
    return cast(Vector3, tuple((bounds[0][index] + bounds[1][index]) * 0.5 for index in range(3)))


def _vector3(value: object, name: str, *, positive: bool) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AttachmentSphereConfigurationError(f"{name} must contain three values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AttachmentSphereConfigurationError(f"{name} must contain real numbers") from error
    if any(not math.isfinite(item) or (positive and item <= 0.0) for item in result):
        qualifier = "finite positive" if positive else "finite"
        raise AttachmentSphereConfigurationError(f"{name} must contain {qualifier} values")
    return cast(Vector3, result)


def _positive_float(value: object, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result <= 0.0:
        raise AttachmentSphereConfigurationError(f"{name} must be positive")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttachmentSphereConfigurationError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AttachmentSphereConfigurationError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result < 0.0:
        raise AttachmentSphereConfigurationError(f"{name} must be finite and non-negative")
    return result


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


_UNIT_DIRECTIONS = tuple(
    (x / length, y / length, z / length)
    for x in (-1.0, 0.0, 1.0)
    for y in (-1.0, 0.0, 1.0)
    for z in (-1.0, 0.0, 1.0)
    if (length := math.sqrt(x * x + y * y + z * z)) > 0.0
)


__all__ = [
    "ALGORITHM_VERSION",
    "AttachmentShapeHint",
    "AttachmentSphereBudgetError",
    "AttachmentSphereConfigurationError",
    "AttachmentTaskProfile",
    "AttachmentValidationStatus",
    "LocalCollisionSphere",
    "StructureAwareSphereRequest",
    "StructureAwareSphereResult",
    "StructureValidation",
    "generate_structure_aware_spheres",
    "parse_structure_aware_request",
    "structure_config_fallback",
    "validate_structure_aware_spheres",
]
