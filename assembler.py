"""Shared finite-element assembly helpers for 1D and 2D scalar problems."""

from __future__ import annotations

import numpy as np


def _as_callable(value):
    if callable(value):
        return value

    constant = float(value)

    def _constant_callable(*_args):
        return constant

    return _constant_callable


def gauss_rule_1d(order):
    """Return Gauss-Legendre points and weights on [-1, 1]."""
    if order != 2:
        raise ValueError("Only 2-point Gauss quadrature is supported in 1D.")

    point = 1.0 / np.sqrt(3.0)
    return [(-point, 1.0), (point, 1.0)]


def gauss_rule_2d(order):
    """Return tensor-product Gauss points and weights on [-1, 1] x [-1, 1]."""
    if order != 2:
        raise ValueError("Only 2x2 Gauss quadrature is supported for quad4 elements.")

    point = 1.0 / np.sqrt(3.0)
    points = [-point, point]
    return [(xi, eta, 1.0) for xi in points for eta in points]


def _shape_functions_1d(xi):
    return np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)


def _shape_gradients_1d(length):
    return np.array([-1.0 / length, 1.0 / length], dtype=float)


def _quad4_shape_functions(xi, eta):
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _quad4_shape_gradients(xi, eta):
    return 0.25 * np.array(
        [
            [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi],
        ],
        dtype=float,
    )


def _tri3_reference_gradients():
    return np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _tri3_quadrature():
    return [
        ((1.0 / 6.0, 1.0 / 6.0), 1.0 / 6.0),
        ((2.0 / 3.0, 1.0 / 6.0), 1.0 / 6.0),
        ((1.0 / 6.0, 2.0 / 3.0), 1.0 / 6.0),
    ]


def _tri3_shape_functions(xi, eta):
    return np.array([1.0 - xi - eta, xi, eta], dtype=float)


def _scatter_matrix(global_matrix, local_matrix, dofs):
    for local_i, global_i in enumerate(dofs):
        for local_j, global_j in enumerate(dofs):
            global_matrix[global_i, global_j] += local_matrix[local_i, local_j]


def _scatter_vector(global_vector, local_vector, dofs):
    for local_i, global_i in enumerate(dofs):
        global_vector[global_i] += local_vector[local_i]


def tri3_element_matrices(coords, problem_data):
    """Return local stiffness and load contributions for a linear triangle."""
    diffusion = _as_callable(problem_data["diffusion"])
    reaction = _as_callable(problem_data.get("reaction", 0.0))
    source = _as_callable(problem_data["source"])

    dN_local = _tri3_reference_gradients()
    jacobian = dN_local @ coords
    det_jacobian = float(np.linalg.det(jacobian))
    if abs(det_jacobian) <= 1e-12:
        raise ValueError("Encountered a degenerate tri3 element with zero area.")

    dN_global = np.linalg.solve(jacobian, dN_local)
    abs_det_jacobian = abs(det_jacobian)
    stiffness = np.zeros((3, 3), dtype=float)
    load = np.zeros(3, dtype=float)

    for (xi, eta), weight in _tri3_quadrature():
        shape_values = _tri3_shape_functions(xi, eta)
        x_gp, y_gp = shape_values @ coords
        k_value = float(diffusion(float(x_gp), float(y_gp)))
        c_value = float(reaction(float(x_gp), float(y_gp)))
        s_value = float(source(float(x_gp), float(y_gp)))

        stiffness += (
            k_value * (dN_global.T @ dN_global) + c_value * np.outer(shape_values, shape_values)
        ) * abs_det_jacobian * weight
        load += shape_values * s_value * abs_det_jacobian * weight

    return stiffness, load


def quad4_element_matrices(coords, problem_data, quadrature_order=2):
    """Return local stiffness and load contributions for a bilinear quad."""
    diffusion = _as_callable(problem_data["diffusion"])
    reaction = _as_callable(problem_data.get("reaction", 0.0))
    source = _as_callable(problem_data["source"])

    stiffness = np.zeros((4, 4), dtype=float)
    load = np.zeros(4, dtype=float)

    for xi, eta, weight in gauss_rule_2d(quadrature_order):
        shape_values = _quad4_shape_functions(xi, eta)
        shape_gradients = _quad4_shape_gradients(xi, eta)
        jacobian = shape_gradients @ coords
        det_jacobian = float(np.linalg.det(jacobian))
        if det_jacobian <= 1e-12:
            raise ValueError("Encountered an invalid quad4 element with non-positive Jacobian.")

        dN_global = np.linalg.solve(jacobian, shape_gradients)
        x_gp, y_gp = shape_values @ coords
        k_value = float(diffusion(float(x_gp), float(y_gp)))
        c_value = float(reaction(float(x_gp), float(y_gp)))
        s_value = float(source(float(x_gp), float(y_gp)))

        stiffness += (
            k_value * (dN_global.T @ dN_global) + c_value * np.outer(shape_values, shape_values)
        ) * det_jacobian * weight
        load += shape_values * s_value * det_jacobian * weight

    return stiffness, load


def assemble(mesh, problem_data, quadrature_order=2):
    """Assemble the 2D global stiffness matrix and load vector."""
    nodes = np.asarray(mesh["nodes"], dtype=float)
    elements = mesh["elements"]

    node_count = len(nodes)
    stiffness = np.zeros((node_count, node_count), dtype=float)
    load = np.zeros(node_count, dtype=float)

    for element in elements:
        element_type = element["type"]
        connectivity = list(element["nodes"])
        coords = nodes[connectivity]

        if element_type == "tri3":
            local_stiffness, local_load = tri3_element_matrices(coords, problem_data)
        elif element_type == "quad4":
            local_stiffness, local_load = quad4_element_matrices(
                coords, problem_data, quadrature_order=quadrature_order
            )
        else:
            raise ValueError(f"Unsupported element type: {element_type}")

        _scatter_matrix(stiffness, local_stiffness, connectivity)
        _scatter_vector(load, local_load, connectivity)

    return {"stiffness": stiffness, "load": load}


def assemble1d(nodes, problem_data, quadrature_order=2, print_level="final"):
    """Assemble the 1D global stiffness matrix, load vector, and element reports."""
    diffusion = _as_callable(problem_data["diffusion"])
    reaction = _as_callable(problem_data.get("reaction", 0.0))
    source = _as_callable(problem_data["source"])

    nodes = np.asarray(nodes, dtype=float)
    node_count = len(nodes)
    stiffness = np.zeros((node_count, node_count), dtype=float)
    load = np.zeros(node_count, dtype=float)
    element_summaries = []
    gauss_points = gauss_rule_1d(quadrature_order)

    for element_index in range(node_count - 1):
        x1 = float(nodes[element_index])
        x2 = float(nodes[element_index + 1])
        length = x2 - x1
        if length <= 0:
            raise ValueError(f"Element {element_index} has non-positive length.")

        jacobian = length / 2.0
        shape_gradients = _shape_gradients_1d(length)
        local_stiffness = np.zeros((2, 2), dtype=float)
        local_load = np.zeros(2, dtype=float)
        gauss_rows = []

        for xi, weight in gauss_points:
            shape_values = _shape_functions_1d(xi)
            x_gp = float(shape_values[0] * x1 + shape_values[1] * x2)
            k_value = float(diffusion(x_gp))
            c_value = float(reaction(x_gp))
            s_value = float(source(x_gp))

            local_stiffness += (
                k_value * np.outer(shape_gradients, shape_gradients)
                + c_value * np.outer(shape_values, shape_values)
            ) * jacobian * weight
            local_load += shape_values * s_value * jacobian * weight

            if print_level == "verbose":
                gauss_rows.append((xi, x_gp, k_value, c_value, s_value))

        connectivity = [element_index, element_index + 1]
        _scatter_matrix(stiffness, local_stiffness, connectivity)
        _scatter_vector(load, local_load, connectivity)
        element_summaries.append(
            {
                "index": element_index,
                "nodes": connectivity,
                "x1": x1,
                "x2": x2,
                "length": length,
                "ke": local_stiffness.copy(),
                "fe": local_load.copy(),
                "gauss_rows": gauss_rows,
                "slope": 0.0,
                "physical_flux": 0.0,
            }
        )

    return stiffness, load, element_summaries
