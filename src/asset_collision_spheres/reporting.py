from .geometry.validation import mesh_fingerprint

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

