"""1-D finite element solver for scalar linear boundary value problems."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

import bvp1d as problem
from assembler import assemble1d
from matrix import matrixsolver
from paths import csv_path
from utils import print_table


TOL = 1e-9


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str
    value: float


def _validate_problem_definition():
    required_attrs = [
        "problem_name",
        "x0",
        "xn",
        "k",
        "c",
        "s",
        "left_bc",
        "right_bc",
        "mesh_mode",
        "base_h",
        "m",
        "h",
        "num_elements",
        "manual_nodes",
        "quadrature_order",
        "print_level",
        "export_csv",
        "plot_result",
    ]
    for attr in required_attrs:
        if not hasattr(problem, attr):
            raise ValueError(f"Missing required problem attribute: {attr}")

    if problem.xn <= problem.x0:
        raise ValueError("xn must be greater than x0.")

    if problem.quadrature_order != 2:
        raise ValueError("Only quadrature_order = 2 is supported in v1.")

    if problem.print_level not in {"stage", "verbose", "final"}:
        raise ValueError("print_level must be one of: stage, verbose, final.")


def _parse_bc(name, raw_bc):
    if not isinstance(raw_bc, dict):
        raise ValueError(f"{name} must be a dictionary.")

    if "type" not in raw_bc or "value" not in raw_bc:
        raise ValueError(f"{name} must contain 'type' and 'value'.")

    kind = str(raw_bc["type"]).strip().lower()
    if kind not in {"dirichlet", "neumann"}:
        raise ValueError(f"{name} type must be 'dirichlet' or 'neumann'.")

    try:
        value = float(raw_bc["value"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} value must be numeric.") from exc

    return BoundaryCondition(kind=kind, value=value)


def _uniform_nodes_from_step(x0, xn, step):
    if step <= 0:
        raise ValueError("Element size must be positive.")

    span = xn - x0
    raw_count = span / step
    count = int(round(raw_count))
    if count <= 0 or abs(raw_count - count) > TOL:
        raise ValueError(
            "The chosen step size does not partition the interval exactly."
        )

    nodes = [x0 + i * step for i in range(count + 1)]
    nodes[-1] = xn
    return nodes


def _build_mesh():
    mode = problem.mesh_mode
    x0 = float(problem.x0)
    xn = float(problem.xn)

    if mode == "m":
        step = float(problem.base_h) / (2 ** int(problem.m))
        nodes = _uniform_nodes_from_step(x0, xn, step)
        return np.array(nodes, dtype=float), {"mode": mode, "step": step}

    if mode == "h":
        step = float(problem.h)
        nodes = _uniform_nodes_from_step(x0, xn, step)
        return np.array(nodes, dtype=float), {"mode": mode, "step": step}

    if mode == "elements":
        element_count = int(problem.num_elements)
        if element_count <= 0:
            raise ValueError("num_elements must be a positive integer.")
        nodes = np.linspace(x0, xn, element_count + 1, dtype=float)
        return nodes, {"mode": mode, "step": (xn - x0) / element_count}

    if mode == "manual":
        if problem.manual_nodes is None:
            raise ValueError("manual_nodes must be provided when mesh_mode='manual'.")
        nodes = np.array(problem.manual_nodes, dtype=float)
        if nodes.ndim != 1 or len(nodes) < 2:
            raise ValueError("manual_nodes must contain at least two coordinates.")
        if abs(nodes[0] - x0) > TOL or abs(nodes[-1] - xn) > TOL:
            raise ValueError("manual_nodes must start at x0 and end at xn.")
        if np.any(np.diff(nodes) <= 0):
            raise ValueError("manual_nodes must be strictly increasing.")
        return nodes, {"mode": mode, "step": None}

    raise ValueError("mesh_mode must be one of: m, h, elements, manual.")


def _get_problem_data():
    raw = getattr(problem, "problemData", {})
    return {
        "diffusion": raw.get("diffusion", problem.k),
        "reaction": raw.get("reaction", problem.c),
        "source": raw.get("source", problem.s),
    }


def _get_boundary_data():
    raw = getattr(problem, "boundaryData", {})
    return raw.get("left", problem.left_bc), raw.get("right", problem.right_bc)


def _format_array(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return "[" + ", ".join(f"{val: .6f}" for val in arr) + "]"
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(f"{val: .6f}" for val in row) + "]")
    return "\n".join(rows)


def _print_matrix(title, matrix):
    print(f"\n{title}:")
    print(_format_array(matrix))


def _print_vector(title, vector):
    print(f"\n{title}:")
    print(_format_array(vector))


def _assemble_system(nodes, print_level):
    return assemble1d(
        nodes,
        _get_problem_data(),
        quadrature_order=problem.quadrature_order,
        print_level=print_level,
    )


def _apply_neumann_bc(force_vector, left_bc, right_bc):
    adjusted = force_vector.copy()
    if left_bc.kind == "neumann":
        adjusted[0] += left_bc.value
    if right_bc.kind == "neumann":
        adjusted[-1] -= right_bc.value
    return adjusted


def _reduce_and_solve(K, F, left_bc, right_bc):
    node_count = len(F)
    prescribed = {}
    if left_bc.kind == "dirichlet":
        prescribed[0] = left_bc.value
    if right_bc.kind == "dirichlet":
        prescribed[node_count - 1] = right_bc.value

    if not prescribed:
        raise ValueError(
            "At least one Dirichlet boundary condition is required for a stable solve."
        )

    all_dofs = list(range(node_count))
    known_dofs = sorted(prescribed)
    free_dofs = [dof for dof in all_dofs if dof not in prescribed]

    u_known = np.array([prescribed[dof] for dof in known_dofs], dtype=float)
    Kuu = K[np.ix_(free_dofs, free_dofs)] if free_dofs else np.zeros((0, 0))
    Fu = F[free_dofs].copy() if free_dofs else np.zeros(0)

    if known_dofs and free_dofs:
        Kuk = K[np.ix_(free_dofs, known_dofs)]
        Fu = Fu - Kuk @ u_known

    full_solution = np.zeros(node_count, dtype=float)
    for dof, value in prescribed.items():
        full_solution[dof] = value

    if free_dofs:
        solved_unknowns = np.array(matrixsolver(Kuu, Fu), dtype=float)
        for index, dof in enumerate(free_dofs):
            full_solution[dof] = solved_unknowns[index]
    else:
        solved_unknowns = np.zeros(0)

    return {
        "known_dofs": known_dofs,
        "free_dofs": free_dofs,
        "u_known": u_known,
        "Kuu": Kuu,
        "Fu": Fu,
        "solution": full_solution,
        "solved_unknowns": solved_unknowns,
    }


def _compute_reactions(K_full, F_full, solution):
    return K_full @ solution - F_full


def _build_mesh_rows(nodes, left_bc, right_bc):
    rows = []
    last_index = len(nodes) - 1
    for idx, x in enumerate(nodes):
        label = f"u_{idx}"
        if idx == 0 and left_bc.kind == "dirichlet":
            label += " (prescribed)"
        elif idx == last_index and right_bc.kind == "dirichlet":
            label += " (prescribed)"
        else:
            label += " (unknown)"
        rows.append((idx, x, label))
    return rows


def _print_problem_summary(nodes, mesh_info, left_bc, right_bc):
    print("=" * 72)
    print("1-D FINITE ELEMENT ANALYSIS")
    print("=" * 72)
    print(f"Problem: {problem.problem_name}")
    print(
        f"Domain: [{problem.x0:.6f}, {problem.xn:.6f}] with {len(nodes) - 1} elements "
        f"and {len(nodes)} nodes"
    )
    print(f"Mesh mode: {mesh_info['mode']}")
    if mesh_info["step"] is not None:
        print(f"Uniform element size: {mesh_info['step']:.6f}")
    print(f"Quadrature order: {problem.quadrature_order}")
    print(f"Print level: {problem.print_level}")
    print(f"Left BC:  {left_bc.kind} = {left_bc.value:.6f}")
    print(f"Right BC: {right_bc.kind} = {right_bc.value:.6f}")


def _print_element_reports(element_summaries):
    for item in element_summaries:
        print("\n" + "-" * 72)
        print(
            f"Element {item['index']} | nodes {item['nodes'][0]}-{item['nodes'][1]} "
            f"| span [{item['x1']:.6f}, {item['x2']:.6f}]"
        )
        _print_matrix("Local stiffness matrix k_e", item["ke"])
        _print_vector("Local load vector f_e", item["fe"])

        if item["gauss_rows"]:
            rows = []
            for xi, x_gp, k_val, c_val, s_val in item["gauss_rows"]:
                rows.append((xi, x_gp, k_val, c_val, s_val))
            print("\nGauss point data:")
            print_table(["xi", "x", "k(x)", "c(x)", "s(x)"], rows)


def _update_element_result_summary(element_summaries, nodes, solution, problem_data):
    diffusion = problem_data["diffusion"]
    for item in element_summaries:
        i, j = item["nodes"]
        length = item["length"]
        slope = (solution[j] - solution[i]) / length
        x_mid = 0.5 * (item["x1"] + item["x2"])
        physical_flux = -float(diffusion(x_mid)) * slope
        item["slope"] = slope
        item["physical_flux"] = physical_flux


def _print_solution_report(
    nodes,
    solution,
    reactions,
    element_summaries,
    exact_values,
    free_dofs,
    known_dofs,
):
    rows = []
    if exact_values is None:
        for idx, (x, value) in enumerate(zip(nodes, solution)):
            rows.append((idx, x, value))
        print("\nNodal solution:")
        print_table(["node", "x", "u(x)"], rows)
    else:
        max_abs_error = 0.0
        for idx, (x, value, exact) in enumerate(zip(nodes, solution, exact_values)):
            error = abs(value - exact)
            max_abs_error = max(max_abs_error, error)
            rows.append((idx, x, value, exact, error))
        print("\nNodal solution and exact-value comparison:")
        print_table(["node", "x", "u(x)", "u_exact(x)", "|error|"], rows)
        print(f"\nMaximum absolute nodal error: {max_abs_error:.6e}")

    unknown_rows = [(f"u_{idx}", solution[idx]) for idx in free_dofs]
    print("\nUnknown values:")
    if unknown_rows:
        print_table(["DOF", "value"], unknown_rows)
    else:
        print("All nodal values are prescribed by Dirichlet boundary conditions.")

    prescribed_rows = [(f"u_{idx}", solution[idx]) for idx in known_dofs]
    if prescribed_rows:
        print("\nPrescribed nodal values:")
        print_table(["DOF", "value"], prescribed_rows)

    reaction_rows = [(f"R_{idx}", value) for idx, value in enumerate(reactions)]
    print("\nResidual / reaction vector (K u - F):")
    print_table(["entry", "value"], reaction_rows)

    element_rows = []
    for item in element_summaries:
        element_rows.append(
            (
                item["index"],
                item["x1"],
                item["x2"],
                item["slope"],
                item["physical_flux"],
            )
        )
    print("\nElement slopes and physical fluxes:")
    print_table(
        ["element", "x_left", "x_right", "du/dx", "-k(x_mid) du/dx"],
        element_rows,
    )


def _write_csv(nodes, solution, exact_values):
    output_path = csv_path("output_fea1d.csv")
    headers = ["node", "x", "u"]
    if exact_values is not None:
        headers.extend(["u_exact", "abs_error"])

    with open(output_path, mode="w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx, (x, value) in enumerate(zip(nodes, solution)):
            row = [idx, f"{x:.6f}", f"{value:.6f}"]
            if exact_values is not None:
                error = abs(value - exact_values[idx])
                row.extend([f"{exact_values[idx]:.6f}", f"{error:.6e}"])
            writer.writerow(row)

    print(f"\nCSV file created: {output_path}")


def _plot_solution(nodes, solution, exact_callable):
    if not problem.plot_result:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nPlot skipped: matplotlib is not available.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(nodes, solution, "o-", color="royalblue", linewidth=1.2, label="FEA solution")

    if exact_callable is not None:
        x_dense = np.linspace(nodes[0], nodes[-1], 400)
        y_dense = [exact_callable(x) for x in x_dense]
        plt.plot(x_dense, y_dense, "--", color="darkorange", linewidth=1.2, label="Exact solution")

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(problem.problem_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def main():
    _validate_problem_definition()

    left_raw_bc, right_raw_bc = _get_boundary_data()
    left_bc = _parse_bc("left_bc", left_raw_bc)
    right_bc = _parse_bc("right_bc", right_raw_bc)
    problem_data = _get_problem_data()
    nodes, mesh_info = _build_mesh()

    _print_problem_summary(nodes, mesh_info, left_bc, right_bc)
    print("\nMesh / DOF table:")
    print_table(["node", "x", "dof"], _build_mesh_rows(nodes, left_bc, right_bc))

    K_full, F_body, element_summaries = _assemble_system(nodes, problem.print_level)
    if problem.print_level in {"stage", "verbose"}:
        _print_element_reports(element_summaries)

    F_full = _apply_neumann_bc(F_body, left_bc, right_bc)

    reduction = _reduce_and_solve(K_full, F_full, left_bc, right_bc)
    solution = reduction["solution"]
    reactions = _compute_reactions(K_full, F_full, solution)
    _update_element_result_summary(element_summaries, nodes, solution, problem_data)

    _print_matrix("Global stiffness matrix K", K_full)
    _print_vector("Global load vector F after Neumann BCs", F_full)
    _print_matrix("Reduced stiffness matrix Kuu", reduction["Kuu"])
    _print_vector("Reduced load vector Fu", reduction["Fu"])
    print(f"\nOrdered unknown DOFs: {reduction['free_dofs']}")
    print(f"Prescribed DOFs: {reduction['known_dofs']}")

    exact_callable = getattr(problem, "exact_solution", None)
    exact_values = None
    if callable(exact_callable):
        exact_values = np.array([float(exact_callable(x)) for x in nodes], dtype=float)

    _print_solution_report(
        nodes,
        solution,
        reactions,
        element_summaries,
        exact_values,
        reduction["free_dofs"],
        reduction["known_dofs"],
    )

    if problem.export_csv:
        _write_csv(nodes, solution, exact_values)

    _plot_solution(nodes, solution, exact_callable if callable(exact_callable) else None)


if __name__ == "__main__":
    main()
