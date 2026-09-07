"""Material-aware collision spheres. All geometry is in metres."""

from .auto_geometry_spheres import generate_auto_geometry_spheres, evaluate_auto_geometry_spheres
from .mesh_region_spheres import generate_mesh_region_spheres, evaluate_mesh_spheres, MeshRegionSphereResult
from .mesh_attachment import check_material_mesh, check_spheres, mesh_fingerprint
from .io import load_mesh, make_report

__version__ = "0.1.0"
__all__ = [
    "generate_auto_geometry_spheres", "evaluate_auto_geometry_spheres",
    "generate_mesh_region_spheres", "evaluate_mesh_spheres", "MeshRegionSphereResult",
    "check_material_mesh", "check_spheres", "mesh_fingerprint", "load_mesh", "make_report",
]
