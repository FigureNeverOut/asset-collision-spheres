"""Material-aware and explicit convex-surface collision spheres, in metres."""

from importlib import import_module

__version__ = "0.1.1"
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


def __getattr__(name):
    # Importing preview.isaac must not load standalone pxr/numerical packages
    # before SimulationApp initializes Kit. Public API names are resolved lazily.
    if name in __all__:
        value = getattr(import_module(".api", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
