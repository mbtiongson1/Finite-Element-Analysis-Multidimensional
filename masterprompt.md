# Master Prompt: 2D FEA Research Kit — Phase Implementation Plans

Use this prompt to generate detailed, executable implementation plans for each phase of the sprint. Replace `{PHASE}` with the phase number and name below.

---

## Template Prompt

**Preamble (context for Claude):**

You are working on a 2D finite element analysis research kit in Python. The project uses:
- `_root_bootstrap.py` for initialization and root-level paths
- `matrix.py` for sparse matrix assembly and solving
- `paths.py` for output directory management
- `utils.py` for shared reporting and visualization helpers
- Current mesh format: triangular and quad elements with node coordinates and connectivity

Your coding style: avoid underscores in function/variable names (except dunder functions). Function outputs are self-contained; no side effects unless explicitly documented.

---

## {PHASE}: {PHASE_NAME}

**Sprint goal:** {GOAL}

**Deliverables:**
- {LIST SPECIFIC FILES}
- {LIST SPECIFIC FUNCTIONS}
- {LIST SPECIFIC TESTS OR BENCHMARKS}

**Success criteria:**
- {CRITERION 1}
- {CRITERION 2}
- {CRITERION 3}

---

### Requirements

**Inputs (from previous phases or existing code):**
- {INPUT 1: description}
- {INPUT 2: description}

**New concepts to implement:**
- {CONCEPT 1}
- {CONCEPT 2}

**Constraints:**
- {CONSTRAINT 1}
- {CONSTRAINT 2}

---

### Detailed Task Breakdown

**Task {N}.{M}: {Task name}**

*Scope:*
- {What this task delivers}

*Implementation notes:*
- {How to approach it}
- {Key decisions}
- {Edge cases to handle}

*Acceptance criteria:*
- Code exists at `{filepath}`
- Function signature: `{signature}`
- Test case: `{test name}`

---

### Integration Checklist

Before marking this phase complete:
- [ ] All new files created in correct locations
- [ ] All functions have docstrings with type hints
- [ ] Tests pass (run `pytest tests/test_{phase_slug}.py`)
- [ ] Output files (if any) are written to `paths.OUTPUT_DIR`
- [ ] Reused components from earlier phases work as expected
- [ ] Code follows style guide (camelCase functions, no underscores)

---

### Files to Create / Modify

| File | Status | Purpose |
|------|--------|---------|
| `{filename}` | new/modify | {purpose} |

---

### Pseudocode Skeleton

Provide high-level pseudocode for the main solver or validation routine:

```python
# {filename}

def {main_function}({inputs}):
    """
    {description}
    
    Args:
        {arg}: {type and description}
    
    Returns:
        {output}: {type and description}
    """
    # Step 1: {step description}
    # Step 2: {step description}
    # ...
    return {output}
```

---

## Actual Phase Specifications

### PHASE 1: Core Solver (P1 Triangle & Q1 Quad Assembly)

**Sprint goal:** Build a working 2D finite element solver for linear scalar PDEs (Poisson, heat, etc.) on triangular and quadrilateral meshes with P1 and Q1 shape functions.

**Deliverables:**
- `solver2d.py` — main solver module
- `assembler.py` — FEA assembly (stiffness + load)
- `boundaryConditions.py` — BC application and enforcement
- `tests/test_solver2d.py` — unit tests
- `benchmarks/poisson_unit_square.py` — reference problem

**Success criteria:**
- Assemble global stiffness matrix for 2D scalar PDE
- Apply Dirichlet and Neumann boundary conditions
- Solve sparse linear system using `matrix.py`
- Produce solution vector u(x, y) at nodes
- Pass verification test against known analytical solution (unit square Poisson)

---

### PHASE 2: Input Validation & Benchmark Verification

**Sprint goal:** Add robust input validation to catch common errors (degenerate meshes, BC over-constraints, isolated nodes) and establish a suite of benchmark problems for verification.

**Deliverables:**
- `validation.py` — mesh and boundary condition checks
- `benchmarks/suite.py` — 2–3 reference problems with known solutions
- `tests/test_validation.py` — validation unit tests
- `tests/test_assembly.py` — assembly correctness tests (stiffness matrix properties, BC assembly, reaction forces)

**Success criteria:**
- Detect and report mesh issues (zero-area elements, collinear nodes, dangling vertices)
- Validate BC consistency (no over-constrained DoFs, solvable system)
- Benchmark: unit square Poisson reproduces analytical solution to within tolerance
- Assembly tests verify: stiffness matrix symmetry, positive definiteness (with Dirichlet), reaction force balance
- All validation errors halt execution with clear messages

---

### PHASE 3A: Post-Processing & Visualization (Contours, Heatmaps, Export)

**Sprint goal:** Add plotting and export functionality so every solve produces publication-ready contour plots and heatmaps saved to `out/`.

**Deliverables:**
- `plotting.py` — contour and heatmap generation (uses matplotlib)
- `export.py` — save plots and metadata to files
- `tests/test_plotting.py` — plot output validation
- Example output files in `out/` directory (timestamped PNG, PDF, JSON metadata)

**Success criteria:**
- Generate contour plot of solution u(x, y) over the 2D domain
- Generate heatmap (color-filled contours) with colorbar
- Save to `out/solution_contour_{timestamp}.png` and `.pdf`
- Include metadata (min/max values, element count, solver time) in JSON sidecar
- Plots render correctly for both uniform and non-uniform meshes
- Handles edge cases (constant solution, negative values)

---

### PHASE 3B: Multi-Material & Regional Source Definitions

**Sprint goal:** Extend the solver to support material properties and source terms that vary by subdomain/region.

**Deliverables:**
- `materials.py` — material property registry and lookup
- `regions.py` — region tagging and management
- `assembler.py` (extended) — element-wise material/source lookup during assembly
- `tests/test_materials.py` — region and material tests
- `benchmarks/composite_heat.py` — composite material reference problem

**Success criteria:**
- Tag mesh elements by region ID during mesh creation
- Store material properties (thermal conductivity, Young's modulus, etc.) per region
- Define source terms (heat generation, body forces) per region
- Assembly respects regional variations: K[e] uses material from element's region, f[e] uses source from region
- Composite heat problem: two materials with different conductivities, correct temperature profile at interface
- Materials and regions are decoupled from mesh structure (same mesh, different problems)

---

### PHASE 4: Higher-Order Shape Functions (P2 Triangles, Serendipity Quads)

**Sprint goal:** Extend the solver to support P2 triangles and serendipity quads for higher accuracy and convergence studies.

**Deliverables:**
- `shapeFunction.py` — P2 and serendipity shape function evaluation and derivatives
- `assembler.py` (extended) — higher-order element integration
- `tests/test_shapeFunctions.py` — shape function tests (partition of unity, gradients)
- `tests/test_convergence.py` — convergence studies (L2 error vs mesh refinement)
- `benchmarks/convergence_poisson.py` — benchmark with analytical solution for convergence plots

**Success criteria:**
- P2 triangles: 6 nodes per element (vertices + edge midpoints), quadratic Lagrange basis
- Serendipity quads: 8 nodes per element (vertices + edge midpoints, no center node)
- Shape functions pass: partition of unity (sum = 1), correct derivatives (gradient test)
- Assembly with higher-order elements produces correct stiffness matrices
- Convergence study: L2 error decreases as O(h³) for P2, matching theory
- All existing Phase 1–3 tests still pass with P1/Q1 (no regression)
- Benchmark problem solved with P1, P2, and serendipity; convergence verified

---

## How to Use This Prompt

1. **For generating Phase N plan:** Copy the section `### PHASE N: {Name}` and the template above it.
2. **Instantiate placeholders:**
   - Replace `{PHASE}` with "1", "2", "3A", "3B", "4"
   - Replace `{PHASE_NAME}` with the full phase name
   - Fill in `{GOAL}`, `{DELIVERABLES}`, `{SUCCESS_CRITERIA}`, etc. with the corresponding section from the phase spec
3. **Request:** "Generate a detailed implementation plan for Phase {N} of the 2D FEA research kit using the master prompt above. Include task breakdown, pseudocode, file structure, and integration checklist."
4. **Output:** Claude will produce a step-by-step, actionable plan with code skeletons, test structure, and clear file organization.

---

## Prompt Customization for Specific Needs

### To emphasize testing:
Add to any phase: "Prioritize test coverage. For every function, write at least one unit test and one integration test. Use pytest fixtures for mesh setup."

### To emphasize documentation:
Add: "Every function must have a docstring with Args, Returns, and Raises sections. Include inline comments for non-obvious logic. Add a 'See also' section linking to related functions."

### To request executable examples:
Add: "For each phase, provide a complete working example script (`examples/{phase_slug}_example.py`) that demonstrates the phase's functionality end-to-end."

### To request performance benchmarks:
Add: "Include a performance profile (time and memory for 1k, 10k, 100k element meshes). Identify bottlenecks and suggest optimization paths."

---

## Example Usage

**User request:**
"Generate the detailed implementation plan for Phase 1: Core Solver using the master prompt. Include pseudocode for the assembly loop, test structure, and file locations."

**Agent's response:**
[Detailed Phase 1 plan with:
- Task breakdown (assembly, BC, solve)
- Pseudocode for `assembler.assemble()`, `boundaryConditions.apply()`, `solver2d.solve()`
- File structure: `solver2d.py`, `assembler.py`, `boundaryConditions.py`, `tests/test_solver2d.py`
- Integration checklist
- Example benchmark problem structure]