"""2-D finite element solver for scalar linear boundary value problems."""

from __future__ import annotations

import numpy as np

import bvp2d as problem
from assembler import assemble
from matrix import matrixsolver


def solve(mesh=None, problemData=None, boundaryData=None, quadratureOrder=None):
    """Solve the active 2D finite-element problem and return all core arrays."""
    active_mesh = mesh if mesh is not None else problem.meshData
    active_problem = problemData if problemData is not None else problem.problemData
    active_boundary = boundaryData if boundaryData is not None else problem.boundaryData
    active_order = quadratureOrder if quadratureOrder is not None else problem.quadrature_order

    assembly = assemble(active_mesh, active_problem, quadrature_order=active_order)
    bc_data = problem.applyBoundaryConditions(
        assembly["stiffness"],
        assembly["load"],
        active_mesh,
        active_boundary,
    )

    solution = np.zeros(len(active_mesh["nodes"]), dtype=float)
    for dof, value in zip(bc_data["prescribedDofs"], bc_data["prescribedValues"]):
        solution[dof] = value

    if bc_data["freeDofs"]:
        solved_unknowns = np.array(
            matrixsolver(bc_data["reducedMatrix"], bc_data["reducedLoad"]),
            dtype=float,
        )
        for index, dof in enumerate(bc_data["freeDofs"]):
            solution[dof] = solved_unknowns[index]

    reactions = assembly["stiffness"] @ solution - bc_data["fullLoad"]
    return {
        "u": solution,
        "stiffness": assembly["stiffness"],
        "load": bc_data["fullLoad"],
        "reactions": reactions,
        "freeDofs": bc_data["freeDofs"],
        "prescribedDofs": bc_data["prescribedDofs"],
        "prescribedValues": bc_data["prescribedValues"],
    }


def _relative_l2_error(nodes, values, exact):
    exact_values = np.array([exact(float(x), float(y)) for x, y in nodes], dtype=float)
    numerator = np.linalg.norm(values - exact_values)
    denominator = np.linalg.norm(exact_values)
    if denominator <= 1e-12:
        return float(numerator)
    return float(numerator / denominator)


def main():
    result = solve()
    print(problem.problem_name)
    print(f"Nodes: {len(problem.meshData['nodes'])}")
    print(f"Elements: {len(problem.meshData['elements'])}")

    exact = getattr(problem, "exact_solution", None)
    if callable(exact):
        error = _relative_l2_error(problem.meshData["nodes"], result["u"], exact)
        print(f"Relative nodal L2 error: {error:.6e}")

    print(f"Unknown DOFs: {len(result['freeDofs'])}")


if __name__ == "__main__":
    main()
