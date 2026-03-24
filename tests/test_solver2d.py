import math

import numpy as np
import pytest

import bvp2d
from assembler import assemble
from solver2d import solve


def _relative_l2_error(nodes, values, exact):
    exact_values = np.array([exact(float(x), float(y)) for x, y in nodes], dtype=float)
    numerator = np.linalg.norm(values - exact_values)
    denominator = np.linalg.norm(exact_values)
    return float(numerator / denominator)


@pytest.fixture
def quadMesh2x2():
    return bvp2d.build_unit_square_quad_mesh(2, 2)


@pytest.fixture
def triMesh2x2():
    return bvp2d.build_unit_square_tri_mesh(2, 2)


@pytest.fixture
def quadMesh4x4():
    return bvp2d.build_unit_square_quad_mesh(4, 4)


@pytest.fixture
def triMesh4x4():
    return bvp2d.build_unit_square_tri_mesh(4, 4)


@pytest.fixture
def quadMesh8x8():
    return bvp2d.build_unit_square_quad_mesh(8, 8)


@pytest.fixture
def triMesh8x8():
    return bvp2d.build_unit_square_tri_mesh(8, 8)


@pytest.fixture
def poissonProblem():
    return {
        "diffusion": lambda x, y: 1.0,
        "reaction": lambda x, y: 0.0,
        "source": lambda x, y: 2.0 * math.pi**2 * math.sin(math.pi * x) * math.sin(math.pi * y),
    }


@pytest.fixture
def zeroDirichletBoundary():
    return {
        "dirichlet": [
            {"tag": "left", "value": 0.0},
            {"tag": "right", "value": 0.0},
            {"tag": "bottom", "value": 0.0},
            {"tag": "top", "value": 0.0},
        ],
        "neumann": [],
    }


@pytest.fixture
def topFluxBoundary():
    return {
        "dirichlet": [
            {"tag": "left", "value": 0.0},
            {"tag": "right", "value": 0.0},
            {"tag": "bottom", "value": 0.0},
        ],
        "neumann": [{"tag": "top", "value": 1.0}],
    }


@pytest.fixture
def exactSolution():
    return lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y)


def test_assemble_quad_matrix_is_square_and_symmetric(quadMesh2x2, poissonProblem):
    assembled = assemble(quadMesh2x2, poissonProblem)
    stiffness = assembled["stiffness"]
    assert stiffness.shape == (len(quadMesh2x2["nodes"]), len(quadMesh2x2["nodes"]))
    assert np.allclose(stiffness, stiffness.T, atol=1e-10)


def test_assemble_tri_matrix_is_square_and_symmetric(triMesh2x2, poissonProblem):
    assembled = assemble(triMesh2x2, poissonProblem)
    stiffness = assembled["stiffness"]
    assert stiffness.shape == (len(triMesh2x2["nodes"]), len(triMesh2x2["nodes"]))
    assert np.allclose(stiffness, stiffness.T, atol=1e-10)


def test_assemble_returns_expected_vector_size_for_quad_mesh(quadMesh2x2, poissonProblem):
    assembled = assemble(quadMesh2x2, poissonProblem)
    assert assembled["load"].shape == (len(quadMesh2x2["nodes"]),)


def test_apply_boundary_conditions_reduces_system_for_dirichlet_case(
    quadMesh2x2, poissonProblem, zeroDirichletBoundary
):
    assembled = assemble(quadMesh2x2, poissonProblem)
    bc_data = bvp2d.applyBoundaryConditions(
        assembled["stiffness"], assembled["load"], quadMesh2x2, zeroDirichletBoundary
    )

    prescribed = set(bc_data["prescribedDofs"])
    expected = {0, 1, 2, 3, 5, 6, 7, 8}
    assert prescribed == expected
    assert bc_data["freeDofs"] == [4]
    assert bc_data["reducedMatrix"].shape == (1, 1)


def test_apply_boundary_conditions_adds_neumann_load_only_on_target_edge(
    quadMesh2x2, poissonProblem, topFluxBoundary
):
    assembled = assemble(quadMesh2x2, poissonProblem)
    bc_data = bvp2d.applyBoundaryConditions(
        assembled["stiffness"], assembled["load"], quadMesh2x2, topFluxBoundary
    )

    added_load = bc_data["fullLoad"] - assembled["load"]
    top_nodes = {6, 7, 8}
    nonzero_nodes = {index for index, value in enumerate(added_load) if abs(value) > 1e-12}
    assert nonzero_nodes == top_nodes
    assert pytest.approx(np.sum(added_load), rel=1e-12, abs=1e-12) == 1.0


def test_apply_boundary_conditions_rejects_missing_dirichlet_constraints(quadMesh2x2, poissonProblem):
    assembled = assemble(quadMesh2x2, poissonProblem)
    with pytest.raises(ValueError, match="Dirichlet"):
        bvp2d.applyBoundaryConditions(
            assembled["stiffness"],
            assembled["load"],
            quadMesh2x2,
            {"dirichlet": [], "neumann": [{"tag": "top", "value": 1.0}]},
        )


def test_apply_boundary_conditions_rejects_conflicting_dirichlet_values(quadMesh2x2, poissonProblem):
    assembled = assemble(quadMesh2x2, poissonProblem)
    with pytest.raises(ValueError, match="Conflicting"):
        bvp2d.applyBoundaryConditions(
            assembled["stiffness"],
            assembled["load"],
            quadMesh2x2,
            {
                "dirichlet": [
                    {"tag": "left", "value": 0.0},
                    {"tag": "bottom", "value": 1.0},
                ],
                "neumann": [],
            },
        )


def test_solve_unit_square_poisson_quad_matches_exact_solution(
    quadMesh8x8, poissonProblem, zeroDirichletBoundary, exactSolution
):
    result = solve(mesh=quadMesh8x8, problemData=poissonProblem, boundaryData=zeroDirichletBoundary)
    error = _relative_l2_error(quadMesh8x8["nodes"], result["u"], exactSolution)
    assert error < 5e-2


def test_solve_unit_square_poisson_tri_matches_exact_solution(
    triMesh8x8, poissonProblem, zeroDirichletBoundary, exactSolution
):
    result = solve(mesh=triMesh8x8, problemData=poissonProblem, boundaryData=zeroDirichletBoundary)
    error = _relative_l2_error(triMesh8x8["nodes"], result["u"], exactSolution)
    assert error < 8e-2


def test_refinement_reduces_error_for_quad_mesh(
    quadMesh4x4,
    quadMesh8x8,
    poissonProblem,
    zeroDirichletBoundary,
    exactSolution,
):
    coarse = solve(mesh=quadMesh4x4, problemData=poissonProblem, boundaryData=zeroDirichletBoundary)
    fine = solve(mesh=quadMesh8x8, problemData=poissonProblem, boundaryData=zeroDirichletBoundary)
    coarse_error = _relative_l2_error(quadMesh4x4["nodes"], coarse["u"], exactSolution)
    fine_error = _relative_l2_error(quadMesh8x8["nodes"], fine["u"], exactSolution)
    assert fine_error < coarse_error


def test_refinement_reduces_error_for_tri_mesh(
    triMesh4x4,
    triMesh8x8,
    poissonProblem,
    zeroDirichletBoundary,
    exactSolution,
):
    coarse = solve(mesh=triMesh4x4, problemData=poissonProblem, boundaryData=zeroDirichletBoundary)
    fine = solve(mesh=triMesh8x8, problemData=poissonProblem, boundaryData=zeroDirichletBoundary)
    coarse_error = _relative_l2_error(triMesh4x4["nodes"], coarse["u"], exactSolution)
    fine_error = _relative_l2_error(triMesh8x8["nodes"], fine["u"], exactSolution)
    assert fine_error < coarse_error
