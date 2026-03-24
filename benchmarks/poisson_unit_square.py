"""Reference benchmark for the unit-square Poisson problem."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

import bvp2d
from solver2d import solve


def _relative_l2_error(nodes, values):
    exact = np.array(
        [bvp2d.exact_solution(float(x), float(y)) for x, y in nodes],
        dtype=float,
    )
    numerator = np.linalg.norm(values - exact)
    denominator = np.linalg.norm(exact)
    return float(numerator / denominator)


def _max_abs_error(nodes, values):
    exact = np.array(
        [bvp2d.exact_solution(float(x), float(y)) for x, y in nodes],
        dtype=float,
    )
    return float(np.max(np.abs(values - exact)))


def run_benchmark(divisions=8):
    """Solve the benchmark problem on structured quad and triangle meshes."""
    quad_mesh = bvp2d.build_unit_square_quad_mesh(divisions, divisions)
    tri_mesh = bvp2d.build_unit_square_tri_mesh(divisions, divisions)
    quad_result = solve(mesh=quad_mesh, problemData=bvp2d.problemData, boundaryData=bvp2d.boundaryData)
    tri_result = solve(mesh=tri_mesh, problemData=bvp2d.problemData, boundaryData=bvp2d.boundaryData)

    return {
        "quad4": {
            "mesh": quad_mesh,
            "result": quad_result,
            "relative_l2_error": _relative_l2_error(quad_mesh["nodes"], quad_result["u"]),
            "max_abs_error": _max_abs_error(quad_mesh["nodes"], quad_result["u"]),
        },
        "tri3": {
            "mesh": tri_mesh,
            "result": tri_result,
            "relative_l2_error": _relative_l2_error(tri_mesh["nodes"], tri_result["u"]),
            "max_abs_error": _max_abs_error(tri_mesh["nodes"], tri_result["u"]),
        },
    }


def main():
    benchmark = run_benchmark()
    print("Unit Square Poisson Benchmark")
    print("element | relative_l2_error | max_abs_error")
    for label in ("quad4", "tri3"):
        row = benchmark[label]
        print(
            f"{label:6s} | {row['relative_l2_error']:.6e} | {row['max_abs_error']:.6e}"
        )


if __name__ == "__main__":
    main()
