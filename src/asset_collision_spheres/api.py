"""Public, simulator-independent generation and file API; all lengths are metres."""

from .contracts import GenerationConfig, SphereResult
from .service import generate, generate_from_arrays, generate_from_file
from .algorithms.auto_geometry import (
    generate_auto_geometry_spheres,
    evaluate_auto_geometry_spheres,
)
from .algorithms.convex_surface import generate_convex_surface_spheres
from .algorithms.geometry_policy import generate_geometry_spheres
from .algorithms.mesh_regions import generate_mesh_region_spheres, evaluate_mesh_spheres
from .geometry.types import MeshRegionSphereResult
from .geometry.validation import check_material_mesh, check_spheres, mesh_fingerprint
from .loaders.mesh import load_mesh
from .reporting import make_report

__all__ = [
    "GenerationConfig",
    "SphereResult",
    "generate",
    "generate_from_arrays",
    "generate_from_file",
    "generate_geometry_spheres",
    "generate_convex_surface_spheres",
    "generate_auto_geometry_spheres",
    "evaluate_auto_geometry_spheres",
    "generate_mesh_region_spheres",
    "evaluate_mesh_spheres",
    "MeshRegionSphereResult",
    "check_material_mesh",
    "check_spheres",
    "mesh_fingerprint",
    "load_mesh",
    "make_report",
]
