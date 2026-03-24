def print_table(headers, rows):
    """
    Print a table nicely formatted for console output.
    Floats are truncated to 6 decimal places.
    """
    print(" | ".join(headers))
    print("-" * (len(headers) * 12))
    for row in rows:
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.6f}")
            else:
                formatted_row.append(str(val))
        print(" | ".join(formatted_row))


def print_table_csv(headers, rows):
    """
    Print a table in CSV format (comma-separated),
    ready for Excel copy-paste.
    Floats formatted to 6 decimals, percentages preserved as strings.
    """
    print(",".join(headers))
    for row in rows:
        formatted_row = []
        for val in row:
            if isinstance(val, str) and val.endswith('%'):
                formatted_row.append(val)
            elif isinstance(val, float):
                formatted_row.append(f"{val:.6f}")
            else:
                formatted_row.append(str(val))
        print(",".join(formatted_row))


def _build_poly_latex(coeffs_np):
    """Return a LaTeX math string for the polynomial defined by coeffs_np (ascending order)."""
    terms = []
    for j, c in enumerate(coeffs_np):
        if j == 0:
            terms.append(f"{c:.4g}")
        elif j == 1:
            terms.append(f"{c:.4g}x")
        else:
            terms.append(f"{c:.4g}x^{{{j}}}")

    if not terms:
        return r"$f(x) = 0$"

    result = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            result += " - " + t[1:]
        else:
            result += " + " + t
    return r"$f(x) = " + result + r"$"


def _annotate_points(ax, xs_np, ys_np):
    """Add y-value labels above each scatter point."""
    for xi, yi in zip(xs_np, ys_np):
        ax.annotate(f"{yi:.4g}", xy=(xi, yi), xytext=(0, 6),
                    textcoords='offset points', ha='center', va='bottom', fontsize=8)


def plot_polynomial(xs, ys, coeffs, label="Polynomial Fit", title="Polynomial Approximation", x_end=None, x_actual=None, y_actual=None):
    """Plot the original data points and fitted polynomial curve.

    xs: array-like, x-coordinates of data points
    ys: array-like, y-coordinates of data points
    coeffs: array-like, polynomial coefficients (a_0, a_1, ..., a_p)
            where polynomial is a_0 + a_1*x + a_2*x^2 + ... + a_p*x^p
    label: label for the curve in the legend
    title: title of the plot
    x_end: if provided, extend the curve to this x value (e.g. xn)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise ImportError("matplotlib and numpy required for plotting.") from e

    xs_np = np.array(xs, dtype=float)
    ys_np = np.array(ys, dtype=float)
    coeffs_np = np.array(coeffs, dtype=float)

    # Create dense x grid for smooth polynomial curve
    x_min = xs_np.min()
    x_max = x_end if x_end is not None else xs_np.max()
    x_dense = np.linspace(x_min, x_max, 400)

    # Evaluate polynomial at dense points: sum(c_j * x^j)
    p_dense = np.polyval(coeffs_np[::-1], x_dense)  # polyval expects descending order

    fig, ax = plt.subplots(figsize=(10, 6))

    # y=0 baseline
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    ax.scatter(xs_np, ys_np, color='blue', s=50, label='Data Points', zorder=5)
    ax.plot(x_dense, p_dense, color='red', linewidth=1, label=label)

    if x_actual is not None and y_actual is not None:
        ax.scatter(np.array(x_actual, dtype=float), np.array(y_actual, dtype=float),
                   color='green', s=60, marker='^', label='Actual', zorder=6)

    _annotate_points(ax, xs_np, ys_np)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.text(0.5, -0.1, _build_poly_latex(coeffs_np), transform=ax.transAxes,
            ha='center', va='top', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(bottom=0.18)
    plt.show()


def plot_polynomials_compare(xs, ys, coeffs_list, labels, title="Polynomial Comparison", x_end=None, x_actual=None, y_actual=None, show_data_points=True):
    """Plot multiple fitted polynomials on the same axes for comparison."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise ImportError("matplotlib and numpy required for plotting.") from e

    xs_np = np.array(xs, dtype=float)
    ys_np = np.array(ys, dtype=float)

    # Create dense x grid for smooth curves
    x_min = xs_np.min()
    x_max = x_end if x_end is not None else xs_np.max()
    x_dense = np.linspace(x_min, x_max, 400)

    n_polys = len(coeffs_list)
    fig, ax = plt.subplots(figsize=(12, 7))

    # y=0 baseline
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    if show_data_points:
        ax.scatter(xs_np, ys_np, color='black', s=60, label='Data Points', zorder=10)

    if x_actual is not None and y_actual is not None:
        ax.scatter(np.array(x_actual, dtype=float), np.array(y_actual, dtype=float),
                   color='green', s=60, marker='^', label='Actual', zorder=11)

    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    poly_lines = []
    for i, (coeffs, lbl) in enumerate(zip(coeffs_list, labels)):
        coeffs_np = np.array(coeffs, dtype=float)
        p_dense = np.polyval(coeffs_np[::-1], x_dense)
        color = colors[i % len(colors)]
        ax.plot(x_dense, p_dense, color=color, linewidth=1, label=lbl)
        poly_lines.append(f"{lbl}:  {_build_poly_latex(coeffs_np)}")

    if show_data_points:
        _annotate_points(ax, xs_np, ys_np)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show each polynomial's LaTeX expression below the axes
    poly_text = "\n".join(poly_lines)
    bottom_margin = 0.08 + 0.05 * n_polys
    ax.text(0.5, -0.1, poly_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, linespacing=1.6)
    fig.subplots_adjust(bottom=bottom_margin + 0.1)
    plt.show()
