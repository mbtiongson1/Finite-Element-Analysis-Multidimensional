"""Wrapper entrypoint for the root-level 1D solver.

Prefer: python -m solver1d
"""

from _root_bootstrap import run


if __name__ == "__main__":
    run("solver1d")
