"""Surface-only graph FPS. No spatial nearest-neighbour edges across air.

Vertices + shared edge midpoints + face centroids + area samples form nodes.
All graph edges lie inside ONE actual triangle. Shortest paths approximate
geodesics from above; coarse/anisotropic triangulations still introduce bias.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import Delaunay, QhullError

from .convex_surface import stats
from .hull_structure import area_samples, project


def build_graph(mesh, seed, sample_count=4096, extra_sites=()):
    sample, sample_faces = area_samples(mesh, sample_count, seed)
    n, e, f = len(mesh.vertices), len(mesh.edges_unique), len(mesh.faces)
    nodes = np.vstack(
        (
            mesh.vertices,
            mesh.vertices[mesh.edges_unique].mean(axis=1),
            mesh.triangles_center,
            sample,
        )
    )
    boundary = np.column_stack(
        (mesh.faces, n + mesh.edges_unique_inverse.reshape(-1, 3))
    )
    # Exact point placement, including the optional grid/feature sites.
    extra_indices = np.empty(0, int)
    if len(extra_sites):
        mapped, _, face_ids = project(mesh, extra_sites)
        extra_indices = np.arange(len(nodes), len(nodes) + len(mapped))
        nodes = np.vstack((nodes, mapped))
        sample_faces = np.r_[sample_faces, face_ids]
    rows, cols = [], []
    for a in range(6):
        for b in range(a + 1, 6):
            rows.append(boundary[:, a])
            cols.append(boundary[:, b])
    # Sample nodes connect to the 6 boundary nodes and centroid of their face.
    links = np.column_stack((boundary, n + e + np.arange(f)))
    ids = np.arange(n + e + f, len(nodes))
    for j in range(7):
        rows.append(ids)
        cols.append(links[sample_faces, j])
    for j in range(6):
        rows.append(n + e + np.arange(f))
        cols.append(boundary[:, j])
    # Samples must not remain isolated leaves on a centroid star: two nearby
    # sites ON THE SAME long triangle would otherwise seem artificially far
    # apart. Add intrinsic 2D triangulation links within that triangle only.
    # This is not a spatial kNN graph: different faces never gain air shortcuts.
    refined, failures = 0, 0
    order = np.argsort(sample_faces, kind="stable")
    ordered_faces = sample_faces[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ordered_faces)) + 1, len(order)]
    for left, right in pairwise(cuts):
        if right - left < 2:
            continue
        face = int(ordered_faces[left])
        local_ids = np.r_[boundary[face], n + e + face, ids[order[left:right]]]
        triangle = mesh.triangles[face]
        tangent = triangle[1] - triangle[0]
        tangent /= np.linalg.norm(tangent)
        axes = np.column_stack((tangent, np.cross(mesh.face_normals[face], tangent)))
        xy = (nodes[local_ids] - triangle[0]) @ axes
        xy, unique = np.unique(xy, axis=0, return_index=True)
        local_ids = local_ids[unique]
        try:
            triangles = Delaunay(xy / max(np.ptp(xy, axis=0))).simplices
        except QhullError:
            failures += 1
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            rows.append(local_ids[triangles[:, a]])
            cols.append(local_ids[triangles[:, b]])
        refined += 1
    # Unique undirected edges: COO duplicate summation would distort distances.
    pairs = np.unique(
        np.sort(np.column_stack((np.concatenate(rows), np.concatenate(cols))), axis=1),
        axis=0,
    )
    length = np.maximum(
        np.linalg.norm(nodes[pairs[:, 0]] - nodes[pairs[:, 1]], axis=1), 1e-15
    )
    row, col = pairs.T
    graph = coo_matrix(
        (np.r_[length, length], (np.r_[row, col], np.r_[col, row])),
        shape=(len(nodes), len(nodes)),
    ).tocsr()
    return (
        graph,
        nodes,
        n + e + f + np.arange(len(sample)),
        sample_faces[: len(sample)],
        extra_indices,
        {
            "intrinsic_face_refinement_count": refined,
            "intrinsic_face_refinement_failures": failures,
        },
    )


def graph_fps(mesh, count, seed, sites=(), kinds=()):
    graph, nodes, reference_ids, reference_faces, extras, refinement = build_graph(
        mesh, seed, extra_sites=sites
    )
    n_components, labels = connected_components(graph, directed=False)
    face_components = labels[mesh.faces[:, 0]]
    area = np.bincount(face_components, weights=mesh.area_faces, minlength=n_components)
    chosen = list(map(int, extras))
    sources = list(kinds)
    nearest = np.full(len(nodes), np.inf)
    for index in chosen:
        nearest = np.minimum(nearest, dijkstra(graph, directed=False, indices=index))
    # Seed each component before refinement (largest area first if cap is tight).
    for component in np.argsort(-area, kind="stable"):
        if len(chosen) >= count:
            break
        candidates = np.flatnonzero(labels == component)
        if not np.isfinite(nearest[candidates]).any():
            distances = np.linalg.norm(
                nodes[candidates] - nodes[candidates].mean(axis=0), axis=1
            )
            index = int(candidates[np.argmax(distances)])
            chosen.append(index)
            sources.append("generic")
            nearest = np.minimum(
                nearest, dijkstra(graph, directed=False, indices=index)
            )
    curve = []
    while len(chosen) < count:
        index = int(np.argmax(nearest))
        if not np.isfinite(nearest[index]) or nearest[index] <= 1e-10:
            break
        chosen.append(index)
        sources.append("generic")
        nearest = np.minimum(nearest, dijkstra(graph, directed=False, indices=index))
        # Exact-coordinate duplicate nodes are not separate physical sites.
        duplicate = np.linalg.norm(nodes - nodes[index], axis=1) <= 1e-12
        nearest[duplicate] = 0
        if len(chosen) % 4 == 0 or len(chosen) == count:
            finite = nearest[reference_ids][np.isfinite(nearest[reference_ids])]
            curve.append({"count": len(chosen), **stats(finite)})
    assigned = np.bincount(labels[chosen], minlength=n_components)
    report = {
        "method": "surface_only_augmented_triangle_graph_dijkstra_fps",
        "graph_node_count": len(nodes),
        "graph_edge_count": graph.nnz // 2,
        "component_count": n_components,
        "missing_components": np.flatnonzero(assigned == 0).tolist(),
        "components": [
            {
                "id": int(i),
                "area_m2": float(area[i]),
                "site_count": int(assigned[i]),
                "reference_graph_distance_m": stats(
                    nearest[reference_ids][
                        (face_components[reference_faces] == i)
                        & np.isfinite(nearest[reference_ids])
                    ]
                ),
            }
            for i in range(n_components)
        ],
        "reference_graph_distance_m": stats(
            nearest[reference_ids][np.isfinite(nearest[reference_ids])]
        ),
        "reference_unreachable_fraction": float(
            np.mean(~np.isfinite(nearest[reference_ids]))
        ),
        "distance_scope": "approximate geodesic upper bound on sampled graph, not exact geodesic or sphere coverage",
        "air_shortcut_edges": 0,
        **refinement,
        "triangulation_caveat": "shared edge midpoints reduce but do not eliminate coarse/anisotropic mesh bias",
    }
    return list(nodes[chosen]), sources, report, curve
