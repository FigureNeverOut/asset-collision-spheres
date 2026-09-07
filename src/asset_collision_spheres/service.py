"""Public facade; delegates unchanged numerical work to the v4/v3 algorithms.

No filesystem discovery, simulator imports, asset downloads or native attachment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import ArrayLike

from .contracts import GenerationConfig, SphereResult

ConfigInput = GenerationConfig | Mapping[str, Any] | None


def _config(config: ConfigInput) -> GenerationConfig:
    if config is None:
        return GenerationConfig()
    if isinstance(config, GenerationConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be GenerationConfig, a mapping, or None")
    raw = dict(config)
    if raw.pop("mode", "auto_geometry") != "auto_geometry":
        raise ValueError(
            "The public facade uses auto_geometry; use legacy APIs for legacy modes"
        )
    allowed = {"geometry_policy", "shape_hint", "max_spheres", "seed"}
    if set(raw) - allowed:
        raise ValueError(
            f"Unknown public configuration keys: {sorted(set(raw) - allowed)}"
        )
    return GenerationConfig(**raw)


def _scale(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("unit_scale_m must be a positive finite number")
    try:
        scale = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("unit_scale_m must be a positive finite number") from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("unit_scale_m must be a positive finite number")
    return scale


def generate(mesh: trimesh.Trimesh, *, config: ConfigInput = None) -> SphereResult:
    """Generate from one triangle mesh ALREADY in the desired axes and metres.

    No centering, rescaling, pose transform, annotation or robot is inferred.
    This new facade defaults to v4 auto; old public entry points keep defaults.
    """
    from .algorithms.geometry_policy import generate_geometry_spheres
    from .geometry.validation import mesh_fingerprint

    request = _config(config)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            "mesh must be a single trimesh.Trimesh; use generate_from_arrays/file"
        )
    result = generate_geometry_spheres(mesh, request.to_dict())
    return SphereResult(
        result.spheres.copy(),
        result.regions,
        result.diagnostics,
        request,
        mesh_fingerprint(mesh),
    )


def generate_from_arrays(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    unit_scale_m: float,
    config: ConfigInput = None,
) -> SphereResult:
    """Generate from [V,3] vertices and zero-based integer [F,3] triangle indices.

    Explicit metres-per-input-unit is mandatory (1 for metres, .001 for mm).
    Coordinates are scaled about their existing origin. Caller arrays are not mutated.
    """
    request, scale = _config(config), _scale(unit_scale_m)
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite [V,3]")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
    ):
        raise ValueError("faces must be nonempty integer [F,3] triangle indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index outside vertex array")
    mesh = trimesh.Trimesh(vertices=vertices * scale, faces=faces.copy(), process=False)
    return generate(mesh, config=request)


def generate_from_file(
    path: str | Path,
    *,
    unit_scale_m: float,
    config: ConfigInput = None,
    mesh_prim: str | None = None,
    up_axis: str | None = None,
) -> SphereResult:
    """Read OBJ/PLY/STL, or one explicit USD Mesh using existing OpenUSD/pxr.

    OBJ/PLY/STL preserve the file axes. USD applies authored transforms and
    converts to Z-up; its result frame is NOT automatically attachment-local.
    No paths are included in the serialized result's source metadata.
    """
    from .loaders.mesh import load_mesh

    request, scale, path = _config(config), _scale(unit_scale_m), Path(path)
    if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        if not mesh_prim:
            raise ValueError("USD input requires an explicit mesh_prim")
        from .loaders.usd import load_material_mesh

        source = {"usd_path": str(path), "mesh_prim": mesh_prim, "unit_scale_m": scale}
        if up_axis is not None:
            source["up_axis"] = up_axis
        try:
            mesh, metadata = load_material_mesh(
                {"source": source}, require_material=False, require_triangles=True
            )
        except ModuleNotFoundError as error:
            if error.name == "pxr":
                raise ImportError(
                    "USD input requires pxr/OpenUSD in the active environment; "
                    "select your existing USD/Isaac environment"
                ) from error
            raise
        metadata.pop("usd_path", None)
        return replace(
            generate(mesh, config=request),
            source=metadata,
            coordinate_frame="usd_stage_transformed_z_up_m",
        )
    if mesh_prim is not None or up_axis is not None:
        raise ValueError("mesh_prim and up_axis apply only to USD inputs")
    mesh = load_mesh(path, unit_scale_m=scale, require_material=False)
    return replace(
        generate(mesh, config=request),
        source={
            "format": path.suffix.lower().lstrip("."),
            "effective_unit_scale_m": scale,
        },
    )
