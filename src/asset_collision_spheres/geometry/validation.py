"""Material mesh and sphere validation, independent of runtime adapters."""
import hashlib
import numpy as np
from .material import material_depth

def characteristic_length(mesh):
    """Rotation/translation-invariant size, also stable under planar subdivision."""
    return float(2 * np.linalg.norm(mesh.vertices - mesh.centroid, axis=1).max())



def sphere_bound_metrics(mesh, spheres, depths):
    expansion = np.maximum(np.asarray(spheres)[:, 3] - depths, 0)
    return {
        "protrusion": float(np.mean(expansion > 1e-8)),
        "protrusion_mean_m": float(expansion.mean()),
        "protrusion_p95_m": float(np.quantile(expansion, 0.95)),
        "volume_ratio": float(np.sum(4 * np.pi / 3 * np.asarray(spheres)[:, 3] ** 3) / mesh.volume),
        "metric_definitions": (
            "protrusion fields describe per-ball expansion bounds, NOT sampled outside volume; "
            "volume_ratio sums ball volumes without overlap correction"
        ),
    }



def mesh_fingerprint(mesh):
    """Object-local identity; tolerate sub-micrometre capture/export rounding only."""
    digest = hashlib.sha256(b"full-material-local-mesh-v1")
    digest.update(np.round(np.asarray(mesh.vertices) / 1e-6).astype("<i8").tobytes())
    digest.update(np.asarray(mesh.faces, dtype="<i8").tobytes())
    return digest.hexdigest()



def check_material_mesh(mesh):
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Material mesh is empty or non-finite")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Material sign is uncertain: require closed, outward-oriented mesh; no hull fallback")
    if np.any(mesh.area_faces <= np.finfo(float).eps * np.linalg.norm(mesh.extents) ** 2):
        raise ValueError("Degenerate material triangles are unsupported")



def check_spheres(mesh, spheres, budget, expansion):
    check_material_mesh(mesh)
    spheres = np.asarray(spheres, dtype=float)
    if spheres.ndim != 2 or spheres.shape[1] != 4 or not 1 <= len(spheres) <= budget:
        raise ValueError("Sphere set must be nonempty Nx4 and within actual native capacity")
    if not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0):
        raise ValueError("Sphere coordinates/radii must be finite and radii positive")
    if isinstance(expansion, bool) or not np.isfinite(expansion) or expansion < 0:
        raise ValueError("max_outward_offset_m must be explicit, finite and nonnegative")
    depths = material_depth(mesh, spheres[:, :3], exact_distance=True)
    # Same 1 µm absolute radius tolerance as native slot readback. In particular
    # do not edit the approved legacy set for a 0.27 µm nearest-query correction.
    tolerance = max(1e-6, characteristic_length(mesh) * 1e-6)
    if np.any(depths <= 0) or np.any(spheres[:, 3] - depths > expansion + tolerance):
        raise ValueError("Invalid material centres or expansion bound; refuse native attachment")
    return depths

