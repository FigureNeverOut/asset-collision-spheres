"""File boundary: explicit units, single mesh, and a simulator-neutral report."""
from pathlib import Path

import numpy as np
import trimesh

from .mesh_attachment import check_material_mesh, mesh_fingerprint


def load_mesh(path, *, unit_scale_m):
    """Load OBJ/PLY/STL without hull fitting; retain the file's coordinate axes.

    Exact duplicate vertices are merged for triangle-soup formats such as STL.
    No hole filling, automatic winding repair, or convex approximation is applied.
    Multi-geometry scenes must be exported as an explicit single material mesh.
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
    check_material_mesh(mesh)
    return mesh


def make_report(mesh, result, config, *, coordinate_frame="input_mesh_axes_m"):
    """Keep coordinates explicit; this is not a native robot-attachment artifact."""
    return {
        "schema": "asset-collision-spheres/1",
        "coordinate_frame": coordinate_frame,
        "units": "m",
        "mesh_sha256": mesh_fingerprint(mesh),
        "generation_config": config,
        "spheres_m": result.spheres.tolist(),
        "sphere_regions": list(result.regions),
        "diagnostics": result.diagnostics,
    }
