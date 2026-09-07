"""Material depth and clearance queries shared by the fitters."""
from itertools import pairwise
import numpy as np
import trimesh

def _closest_point_exact(mesh, points):
    """Minimum triangle distance, without proximity's normal-based near-tie swap.

    Trimesh's generic closest_point can replace the closest face by a nearby one
    using an absolute squared-distance tolerance. In millimetre-thick walls that
    changes depth after a rotation/export. Normalize each triangle query and
    retain the actual minimum. Legacy manual generation keeps its old default.
    """
    candidates = trimesh.proximity.nearby_faces(mesh, points)
    lengths = np.array([len(row) for row in candidates])
    faces = np.concatenate(candidates)
    rows = np.repeat(np.arange(len(points)), lengths)
    triangles = np.asarray(mesh.triangles[faces])
    origins = triangles[:, :1, :]
    relative = triangles - origins
    scales = np.linalg.norm(relative, axis=2).max(axis=1)
    if np.any(scales <= 0):
        raise ValueError("Degenerate triangle in exact material query")
    closest_relative = trimesh.triangles.closest_point(
        relative / scales[:, None, None], (points[rows] - origins[:, 0]) / scales[:, None]
    ) * scales[:, None]
    closest = closest_relative + origins[:, 0]
    distances = np.einsum("ij,ij->i", points[rows]-closest, points[rows]-closest)
    starts = np.r_[0, np.cumsum(lengths)]
    chosen = np.array([start + np.argmin(distances[start:end]) for start, end in pairwise(starts)])
    return closest[chosen], np.sqrt(distances[chosen]), faces[chosen]



def material_depth(mesh, points, *, exact_distance=False):
    """Positive inside, negative outside; angle-weighted closest-feature sign.

    A single parity ray can misclassify the scan cup's empty cavity when it hits
    mesh edges. Use face / summed edge / angle-weighted vertex pseudonormals.
    Reference: Baerentzen & Aanaes, TVCG 2005, doi:10.1109/TVCG.2005.49.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty(0)
    vertex_normals = trimesh.geometry.weighted_vertex_normals(
        len(mesh.vertices), mesh.faces, mesh.face_normals, mesh.face_angles
    )
    edge_normals = np.zeros((len(mesh.edges_unique), 3))
    np.add.at(edge_normals, mesh.edges_unique_inverse, np.repeat(mesh.face_normals, 3, axis=0))
    face_edges = mesh.edges_unique_inverse.reshape(-1, 3)
    output = []
    for chunk in np.array_split(points, max(1, (len(points) + 1023) // 1024)):
        query = _closest_point_exact if exact_distance else trimesh.proximity.closest_point
        closest, distance, face = query(mesh, chunk)
        barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[face], closest)
        if not np.isfinite(barycentric).all():
            raise ValueError("Degenerate closest triangle in mesh distance query")
        boundary = barycentric < 1e-7
        boundary_count = boundary.sum(axis=1)
        normal = mesh.face_normals[face].copy()
        at_vertex = boundary_count >= 2
        corner = np.argmax(barycentric[at_vertex], axis=1)
        normal[at_vertex] = vertex_normals[mesh.faces[face[at_vertex], corner]]
        at_edge = boundary_count == 1
        opposite_corner = np.argmax(boundary[at_edge], axis=1)
        edge = face_edges[face[at_edge], (opposite_corner + 1) % 3]
        normal[at_edge] = edge_normals[edge]
        dot = np.einsum("ij,ij->i", chunk - closest, normal)
        if np.any((np.abs(dot) < 1e-14) & (distance > 1e-7)):
            raise ValueError("Ambiguous pseudonormal sign; cannot certify material depth")
        output.append(np.where(dot < 0, distance, -distance))
    return np.concatenate(output)



def sphere_clearance(points, spheres, probe_radius=0.0):
    points, spheres = np.asarray(points, dtype=float), np.asarray(spheres, dtype=float)
    if not len(spheres):
        return np.full(len(points), np.inf)
    return np.concatenate(
        [
            (np.linalg.norm(chunk[:, None, :] - spheres[None, :, :3], axis=2) - spheres[None, :, 3]).min(1)
            - probe_radius
            for chunk in np.array_split(points, max(1, (len(points) + 1023) // 1024))
        ]
    )

