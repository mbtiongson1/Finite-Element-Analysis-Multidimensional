import numpy as np


def matrixsolver(A, C):
    """
    Solves the linear system [A]{y} = {C} for the coefficient vector {y}.

    Parameters:
        A : array-like, shape (n, n) — coefficient matrix
        C : array-like, shape (n,)   — right-hand side vector

    Returns:
        y : numpy array of solution coefficients
    """
    A = np.array(A, dtype=float)
    C = np.array(C, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    if C.ndim != 1 or C.shape[0] != A.shape[0]:
        raise ValueError(f"C must be a vector of length {A.shape[0]}, got shape {C.shape}")

    y = np.linalg.solve(A, C)
    return y
