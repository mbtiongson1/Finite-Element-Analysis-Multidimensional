# Bootstrap Wrapper

This folder now only keeps the bootstrap helper and the optional wrapper for the root-level solver.

## Active Entry Point

- `python root/solver1d.py`

## Notes

- `root/_root_bootstrap.py` stays here so future wrappers can reuse the same launch path.
- Preferred direct execution is:
  - `python -m solver1d`
