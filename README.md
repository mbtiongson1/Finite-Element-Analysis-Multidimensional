# Root-Level 1D Finite Element Solver

This repository contains a flattened root-level finite element analysis workflow centered on a 1D scalar boundary value solver. The current implementation solves linear second-order problems of the form:

```text
-(k(x) u'(x))' + c(x) u(x) = s(x)
```

on a 1D domain with configurable boundary conditions, mesh controls, CSV export, and optional plotting.

## Current Scope

- 1D finite element solver only
- Scalar linear boundary value problems
- Two-node linear elements
- 2-point Gauss quadrature
- Dirichlet and Neumann boundary conditions
- Uniform or manually specified meshes
- Optional exact-solution comparison
- CSV output and optional matplotlib plots

## Repository Layout

```text
.
├── solver1d.py            # Main 1D solver entrypoint
├── bvp1d.py               # Problem definition and solver settings
├── matrix.py              # Linear system solver helper
├── paths.py               # Output-path utilities
├── utils.py               # Console/plot helper utilities
├── root/
│   ├── _root_bootstrap.py # Wrapper bootstrap for future entrypoints
│   └── solver1d.py        # Optional wrapper for the root-level solver
├── out/
│   └── csv/               # Generated CSV outputs
└── docs/
```

## Requirements

The active runtime dependency set is small:

- `numpy` is required
- `matplotlib` is optional, but needed if `plot_result = True`

Standard-library modules already used by the project and do not need installation:

- `csv`
- `dataclasses`
- `pathlib`

Recommended environment:

- Python 3.9+

Install dependencies with:

```bash
pip install numpy matplotlib
```

If you do not need plots, `numpy` alone is sufficient:

```bash
pip install numpy
```

## Running the Solver

Preferred direct execution:

```bash
python -m solver1d
```

Optional wrapper execution:

```bash
python root/solver1d.py
```

The wrapper is kept so the bootstrap mechanism under `root/_root_bootstrap.py` remains available for future solvers and tools.

## Problem Configuration

Edit `bvp1d.py` to define the problem being solved.

### Governing Functions

You provide:

- `k(x)` for the diffusion or stiffness coefficient
- `c(x)` for the reaction coefficient
- `s(x)` for the source term

### Boundary Conditions

Boundary conditions are defined as dictionaries:

```python
left_bc = {"type": "dirichlet", "value": 0.0}
right_bc = {"type": "neumann", "value": 1.0}
```

Supported types:

- `dirichlet`
- `neumann`

At least one Dirichlet boundary condition is required for a stable solve.

### Mesh Controls

Available mesh modes:

- `"m"`: uses `base_h / (2 ** m)`
- `"h"`: uses explicit `h`
- `"elements"`: uses explicit `num_elements`
- `"manual"`: uses `manual_nodes`

Examples:

```python
mesh_mode = "elements"
num_elements = 8
```

```python
mesh_mode = "manual"
manual_nodes = [0.0, 0.1, 0.2, 0.5, 1.0]
```

### Output and Reporting Controls

These settings are also defined in `bvp1d.py`:

- `quadrature_order`
- `print_level`
- `export_csv`
- `plot_result`
- `exact_solution`

Current supported values:

- `quadrature_order = 2` only
- `print_level = "stage" | "verbose" | "final"`

## Example Configuration

```python
problem_name = "Sample 1-D Poisson BVP"

x0 = 0.0
xn = 1.0

def k(x):
    return 1.0

def c(x):
    return 0.0

def s(x):
    return 1.0

left_bc = {"type": "dirichlet", "value": 0.0}
right_bc = {"type": "dirichlet", "value": 0.0}

mesh_mode = "elements"
num_elements = 8

quadrature_order = 2
print_level = "verbose"
export_csv = True
plot_result = True

def exact_solution(x):
    return 0.5 * x * (1.0 - x)
```

## Solver Output

During execution, the solver can print:

- problem summary
- mesh and degree-of-freedom table
- local element matrices and vectors
- Gauss-point data
- global assembled system
- reduced system for unknown DOFs
- nodal solution table
- exact-solution error table, if provided
- reaction/residual vector
- element slopes and fluxes

If `export_csv = True`, the solver writes:

- `out/csv/output_fea1d.csv`

If `plot_result = True` and `matplotlib` is installed, the solver also opens a plot comparing:

- FEA nodal solution
- exact solution, if available

## Implementation Notes

The current solver in `solver1d.py` does the following:

- validates the problem definition in `bvp1d.py`
- builds the mesh from the selected mesh mode
- assembles the global stiffness matrix and load vector
- applies Neumann contributions to the force vector
- reduces the system using prescribed Dirichlet DOFs
- solves the reduced linear system
- reconstructs the full nodal solution
- computes reactions and element-level derived quantities
- optionally exports CSV data and plots results

## Known Limitations

- Only 1D problems are implemented
- Only linear two-node elements are implemented
- Only 2-point Gauss quadrature is supported
- The CSV filename still uses the legacy name `output_fea1d.csv`
- Plotting currently focuses on a simple 1D line result view

## Future Enhancements

Planned or recommended next steps:

- Add a 2D finite element solver for triangular and quadrilateral meshes
- Add a 3D finite element solver for tetrahedral and hexahedral meshes
- Support richer element formulations and higher-order shape functions
- Add material and source definitions that vary by region or subdomain
- Improve post-processing and visual artefacts, including:
- higher-quality mesh rendering
- deformed-shape visualization
- contour and heat-map plots for 2D fields
- surface and slice plots for 3D fields
- stress/flux overlays and cleaner legends/colorbars
- export plots and reports directly to files under `out/`
- Add input validation for more physical and numerical edge cases
- Add automated tests for mesh generation, assembly, and boundary-condition handling
- Add benchmark problems for 1D, 2D, and 3D verification

## Practical Next Step

The cleanest next extension is to keep the current root-level structure and add parallel entrypoints such as:

- `solver2d.py`
- `solver3d.py`
- `bvp2d.py`
- `bvp3d.py`

while continuing to reuse:

- `root/_root_bootstrap.py`
- `matrix.py`
- `paths.py`
- shared reporting and plotting helpers in `utils.py`
