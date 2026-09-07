"""File boundary: explicit units, single mesh, and a simulator-neutral report."""
from pathlib import Path

import numpy as np
import trimesh

from ..geometry.validation import check_material_mesh


def load_mesh(path, *, unit_scale_m, require_material=True):
    """Load OBJ/PLY/STL without hull fitting; retain the file's coordinate axes.

    Exact duplicate vertices are merged for triangle-soup formats such as STL.
    No hole filling, automatic winding repair, or convex approximation is applied.
    Multi-geometry scenes must be exported as an explicit single material mesh.
    For convex_surface, require_material=False bypasses only the final material
    validity check; units and file geometry handling remain unchanged.
    """
    if isinstance(unit_scale_m, bool) or not np.isfinite(unit_scale_m) or unit_scale_m <= 0:
        raise ValueError("unit_scale_m must be finite and positive")
    path = Path(path)
    if path.suffix.lower() not in {".obj", ".ply", ".stl"}:
        raise ValueError("Supported mesh files: OBJ, PLY, STL; use --preset for USD")
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected one triangle mesh; export scene geometry explicitly")
    # Restore shared indices without rounding or otherwise repairing the surface.
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
    mesh.apply_scale(float(unit_scale_m))
    if require_material:
        check_material_mesh(mesh)
    return mesh
