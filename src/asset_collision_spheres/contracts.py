"""Typed public request/response contracts; lengths and radii are metres."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

GeometryPolicy = Literal["auto", "convex_hull", "original_surface"]
ShapeHint = Literal["auto", "open_box", "box", "generic"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Small v4 configuration. The cap is not a promise of collision coverage."""

    geometry_policy: GeometryPolicy = "auto"
    shape_hint: ShapeHint = "auto"
    max_spheres: int = 64
    seed: int = 5

    def __post_init__(self) -> None:
        from .algorithms.geometry_policy import normalize

        normalize(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"mode": "auto_geometry", **asdict(self)}


@dataclass(frozen=True, slots=True)
class SphereResult:
    """Portable result, separate from any robot/native attachment state.

    ``spheres`` is float64 [N,4] = [x,y,z,r], expressed in ``coordinate_frame``.
    The array/dictionaries remain editable; dataclass fields are read-only.
    """

    spheres: NDArray[np.float64]
    regions: tuple[str, ...]
    diagnostics: dict[str, Any]
    config: GenerationConfig
    mesh_sha256: str
    coordinate_frame: str = "input_mesh_axes_m"
    source: dict[str, Any] | None = None

    @property
    def count(self) -> int:
        return len(self.spheres)

    @property
    def selected_policy(self) -> GeometryPolicy:
        return self.diagnostics["geometry_policy_selected"]

    @property
    def review_required(self) -> bool:
        return bool(self.diagnostics["review_required"])

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible report using the existing schema."""
        report = {
            "schema": "asset-collision-spheres/1",
            "coordinate_frame": self.coordinate_frame,
            "units": "m",
            "mesh_sha256": self.mesh_sha256,
            "generation_config": self.config.to_dict(),
            "spheres_m": self.spheres.tolist(),
            "sphere_regions": list(self.regions),
            "diagnostics": self.diagnostics,
        }
        if self.source is not None:
            report["source"] = self.source
        return json.loads(json.dumps(report, allow_nan=False))

    def save_json(self, path: str | Path) -> Path:
        """Save a portable report; refuse to overwrite an existing file."""
        path = Path(path)
        serialized = (
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(serialized)
        return path
