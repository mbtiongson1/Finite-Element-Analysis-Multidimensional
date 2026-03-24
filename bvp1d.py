"""1-D boundary value problem configuration for the root-level solver."""

problem_name = "Sample 1-D Poisson BVP"

# Domain [x0, xn]
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

boundaryData = {"left": left_bc, "right": right_bc}


mesh_mode = "elements"
base_h = 0.25
m = 0
h = 0.25
num_elements = 8
manual_nodes = None


quadrature_order = 2
print_level = "verbose"
export_csv = True
plot_result = True

problemData = {
    "diffusion": k,
    "reaction": c,
    "source": s,
}


def exact_solution(x):
    return 0.5 * x * (1.0 - x)
