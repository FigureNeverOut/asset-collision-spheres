"""Small explicit regression geometries, no semantic data used by the router."""

import numpy as np
import trimesh


def cells_mesh(cells, pitch=0.025):
    cells = set(cells)
    vertices, faces = [], []
    # Only exposed faces of a union of grid cells; no overlapping solids.
    for cell in sorted(cells):
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for sign in (-1, 1):
                neighbour = list(cell)
                neighbour[axis] += sign
                if tuple(neighbour) in cells:
                    continue
                corners = []
                for u, v in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                    point = np.asarray(cell, float)
                    point[axis] += sign > 0
                    point[others] += [u, v]
                    corners.append(point * pitch)
                normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                if normal[axis] * sign < 0:
                    corners.reverse()
                offset = len(vertices)
                vertices.extend(corners)
                faces.extend(
                    [[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]]
                )
    return trimesh.Trimesh(vertices, faces, process=True)


def synthetic_case(name):
    if name in {"solid_box", "subdivided_box", "rounded_box", "irregular_scan"}:
        mesh = trimesh.creation.box(extents=[0.24, 0.18, 0.12])
        if name == "subdivided_box":
            mesh = mesh.subdivide().subdivide()
        if name == "rounded_box":
            mesh = trimesh.creation.icosphere(subdivisions=3)
            p = mesh.vertices
            mesh.vertices = (
                np.sign(p) * np.abs(p) ** 0.35 * np.array([0.12, 0.09, 0.06])
            )
        if name == "irregular_scan":
            mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        return mesh
    if name == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    if name == "straight_rod":
        return trimesh.creation.cylinder(radius=0.012, height=0.35, sections=32)
    if name in {"u_solid", "l_solid", "branched_solid"}:
        cells = [
            (x, y, 0)
            for x in range(7)
            for y in range(10)
            if (
                y == 0
                or x == 0
                or (name == "u_solid" and x == 6)
                or (name == "branched_solid" and y in (4, 9))
            )
        ]
        return cells_mesh(cells)
    if name in {"curved_rod", "hollow_bent_tube", "ring_solid"}:
        # Partial/full torus about Z. Bent tube has an annular end section.
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        radius = 0.018
        outer = np.column_stack(
            (0.12 + radius * np.cos(angles), radius * np.sin(angles))
        )
        if name == "hollow_bent_tube":
            # trimesh revolve caps may need optional triangulator: form each
            # swept section and end annulus directly instead.
            inner = np.column_stack(
                (0.12 + 0.011 * np.cos(angles), 0.011 * np.sin(angles))
            )
            profile = np.vstack((outer, inner))
        else:
            profile = outer
        closed = name == "ring_solid"
        theta = np.linspace(
            0, 2 * np.pi if closed else 1.45 * np.pi, 64, endpoint=not closed
        )
        vertices = np.array(
            [[r * np.cos(t), r * np.sin(t), z] for t in theta for r, z in profile]
        )
        width, faces = len(profile), []
        for i in range(len(theta) if closed else len(theta) - 1):
            j = (i + 1) % len(theta)
            for ring in range(width // 24):
                for k in range(24):
                    a = i * width + ring * 24 + k
                    b = i * width + ring * 24 + (k + 1) % 24
                    c = j * width + ring * 24 + (k + 1) % 24
                    d = j * width + ring * 24 + k
                    tri = [[a, b, c], [a, c, d]]
                    faces.extend([t[::-1] for t in tri] if ring else tri)
        if not closed:
            for end in [0, len(theta) - 1]:
                offset = end * width
                if width == 48:
                    for k in range(24):
                        a, b = offset + k, offset + (k + 1) % 24
                        faces.extend([[a, b, b + 24], [a, b + 24, a + 24]])
                else:
                    for k in range(1, 23):
                        faces.append([offset, offset + k, offset + k + 1])
        mesh = trimesh.Trimesh(vertices, faces, process=True)
        mesh.fix_normals()
        return mesh
    if name == "narrow_neck":
        profile = [
            [0, 0],
            [0.09, 0],
            [0.09, 0.12],
            [0.02, 0.17],
            [0.02, 0.2],
            [0.014, 0.2],
            [0.014, 0.17],
            [0.084, 0.116],
            [0.084, 0.008],
            [0, 0.008],
        ]
        return trimesh.creation.revolve(profile, sections=64)
    raise ValueError(name)
