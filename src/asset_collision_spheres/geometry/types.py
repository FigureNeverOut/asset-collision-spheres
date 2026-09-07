from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class MeshRegionSphereResult:
    spheres: np.ndarray
    regions: tuple[str, ...]
    diagnostics: dict[str, Any]

