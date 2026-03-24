"""2-D benchmark problem definitions and boundary-condition helpers for solver2d."""

from __future__ import annotations

import math

import numpy as np


def _node_index(i, j, nx):
    return j * (nx + 1) + i


def build_unit_square_quad_mesh(nx, ny):
    """Return a structured quad4 mesh on the unit square."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")

    nodes = []
    for j in range(ny + 1):
        y = j / ny
        for i in range(nx + 1):
            x = i / nx
            nodes.append((x, y))

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = _node_index(i, j, nx)
            n1 = _node_index(i + 1, j, nx)
            n2 = _node_index(i + 1, j + 1, nx)
            n3 = _node_index(i, j + 1, nx)
            elements.append({"type": "quad4", "nodes": [n0, n1, n2, n3]})

    boundary_edges = []
    for j in range(ny):
        boundary_edges.append(
            {
                "tag": "left",
                "nodes": [_node_index(0, j, nx), _node_index(0, j + 1, nx)],
                "normal": (-1.0, 0.0),
            }
        )
        boundary_edges.append(
            {
                "tag": "right",
                "nodes": [_node_index(nx, j, nx), _node_index(nx, j + 1, nx)],
                "normal": (1.0, 0.0),
            }
        )
    for i in range(nx):
        boundary_edges.append(
            {
                "tag": "bottom",
                "nodes": [_node_index(i, 0, nx), _node_index(i + 1, 0, nx)],
                "normal": (0.0, -1.0),
            }
        )
        boundary_edges.append(
            {
                "tag": "top",
                "nodes": [_node_index(i, ny, nx), _node_index(i + 1, ny, nx)],
                "normal": (0.0, 1.0),
            }
        )

    return {
        "nodes": np.array(nodes, dtype=float),
        "elements": elements,
        "boundaryEdges": boundary_edges,
    }


def build_unit_square_tri_mesh(nx, ny):
    """Return a structured tri3 mesh on the unit square."""
    mesh = build_unit_square_quad_mesh(nx, ny)
    tri_elements = []

    for element in mesh["elements"]:
        n0, n1, n2, n3 = element["nodes"]
        tri_elements.append({"type": "tri3", "nodes": [n0, n1, n2]})
        tri_elements.append({"type": "tri3", "nodes": [n0, n2, n3]})

    return {
        "nodes": mesh["nodes"].copy(),
        "elements": tri_elements,
        "boundaryEdges": list(mesh["boundaryEdges"]),
    }


def diffusion(x, y):
    return 1.0


def reaction(x, y):
    return 0.0


def source(x, y):
    return 2.0 * math.pi**2 * math.sin(math.pi * x) * math.sin(math.pi * y)


def exact_solution(x, y):
    return math.sin(math.pi * x) * math.sin(math.pi * y)


problem_name = "Unit Square Poisson Benchmark"
quadrature_order = 2
meshData = build_unit_square_quad_mesh(4, 4)
problemData = {
    "diffusion": diffusion,
    "reaction": reaction,
    "source": source,
}
boundaryData = {
    "dirichlet": [
        {"tag": "left", "value": 0.0},
        {"tag": "right", "value": 0.0},
        {"tag": "bottom", "value": 0.0},
        {"tag": "top", "value": 0.0},
    ],
    "neumann": [],
}


def _evaluate_boundary_value(value, x, y, nx=None, ny=None):
    if not callable(value):
        return float(value)

    attempts = [(x, y, nx, ny), (x, y), (x,), tuple()]
    last_error = None
    for args in attempts:
        try:
            return float(value(*args))
        except TypeError as exc:
            last_error = exc
    raise TypeError("Boundary value callable has an unsupported signature.") from last_error


def _edge_shape_functions(xi):
    return np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)


def _edge_quadrature():
    point = 1.0 / np.sqrt(3.0)
    return [(-point, 1.0), (point, 1.0)]


def integrate_neumann_edge(edge_coords, value, normal=None):
    """Return the equivalent nodal load vector for a boundary edge."""
    edge_coords = np.asarray(edge_coords, dtype=float)
    tangent = edge_coords[1] - edge_coords[0]
    length = float(np.linalg.norm(tangent))
    if length <= 1e-12:
        raise ValueError("Boundary edge has zero length.")

    if normal is None:
        normal = np.array([tangent[1], -tangent[0]], dtype=float)
        normal /= np.linalg.norm(normal)
    else:
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)

    load = np.zeros(2, dtype=float)
    jacobian = length / 2.0

    for xi, weight in _edge_quadrature():
        shape_values = _edge_shape_functions(xi)
        x_gp, y_gp = shape_values @ edge_coords
        flux = _evaluate_boundary_value(value, float(x_gp), float(y_gp), normal[0], normal[1])
        load += shape_values * flux * jacobian * weight

    return load


def applyBoundaryConditions(K, f, mesh, boundary_data):
    """Apply Dirichlet and Neumann boundary conditions for solver2d."""
    full_load = np.asarray(f, dtype=float).copy()
    prescribed = {}

    for edge in mesh.get("boundaryEdges", []):
        tag = edge["tag"]
        edge_nodes = list(edge["nodes"])
        edge_coords = np.asarray(mesh["nodes"][edge_nodes], dtype=float)
        edge_normal = edge.get("normal")

        for bc in boundary_data.get("neumann", []):
            if bc["tag"] != tag:
                continue
            edge_load = integrate_neumann_edge(edge_coords, bc["value"], normal=edge_normal)
            for local_index, global_index in enumerate(edge_nodes):
                full_load[global_index] += edge_load[local_index]

        for bc in boundary_data.get("dirichlet", []):
            if bc["tag"] != tag:
                continue
            for node_id in edge_nodes:
                x, y = mesh["nodes"][node_id]
                value = _evaluate_boundary_value(bc["value"], float(x), float(y))
                if node_id in prescribed and not np.isclose(prescribed[node_id], value):
                    raise ValueError("Conflicting Dirichlet boundary conditions")
                prescribed[node_id] = value

    if not prescribed:
        raise ValueError("At least one Dirichlet boundary condition is required")

    all_dofs = list(range(len(full_load)))
    prescribed_dofs = sorted(prescribed)
    free_dofs = [dof for dof in all_dofs if dof not in prescribed]
    prescribed_values = np.array([prescribed[dof] for dof in prescribed_dofs], dtype=float)
    reduced_matrix = K[np.ix_(free_dofs, free_dofs)] if free_dofs else np.zeros((0, 0))
    reduced_load = full_load[free_dofs].copy() if free_dofs else np.zeros(0)

    if free_dofs and prescribed_dofs:
        coupling = K[np.ix_(free_dofs, prescribed_dofs)]
        reduced_load -= coupling @ prescribed_values

    return {
        "fullLoad": full_load,
        "reducedMatrix": reduced_matrix,
        "reducedLoad": reduced_load,
        "freeDofs": free_dofs,
        "prescribedDofs": prescribed_dofs,
        "prescribedValues": prescribed_values,
    }
