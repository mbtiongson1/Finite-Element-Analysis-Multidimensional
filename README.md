# Root-Level 1D Solver

This repository is now a flattened root-level 1D finite element solver.

## Active Files

- `solver1d.py`
- `bvp1d.py`
- `matrix.py`
- `paths.py`
- `utils.py`
- `root/_root_bootstrap.py`
- `root/solver1d.py`

## Run

Preferred direct execution:

```bash
python -m solver1d
```

Optional wrapper:

```bash
python root/solver1d.py
```

The solver writes CSV output under `out/csv/` when `export_csv = True` in `bvp1d.py`.
