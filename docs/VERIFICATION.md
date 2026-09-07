# Extraction verification — 2026-09-07

Python 3.12 on Linux; dependency versions are recorded in `constraints-tested.txt`.
The dedicated `.venv` does not expose system site-packages. Dependencies were
installed from existing pip wheel caches after a slow network download was stopped.
No original FastSim, cuRobo, Isaac Sim, task2sim or Torch package is installed there.

Results:

- Four imported algorithm suites: **67 passed** (existing simulation environment,
  with imports redirected to this package, before isolated installation).
- Entire standalone suite: **78 passed in 5.37s** in the dedicated environment.
- Installed `asset-spheres` entry point tested from a temporary directory outside
  the repository: valid JSON report and refusal to overwrite an existing file.
- OBJ, PLY and STL load tests: explicit millimetre-to-metre conversion and closed
  material topology preserved.
- Optional OpenUSD test: authored translation, explicit unit override, Y-up to
  Z-up conversion, and physical dimension mismatch rejection.
- Core import test: no FastSim/cuRobo/Isaac/OpenUSD/Torch modules loaded.
- Synthetic cup example: **16 spheres** generated using `demo_auto.json`.
- `python -m pip check`: **No broken requirements found**.
- `python -m build --no-isolation`: source distribution and Python wheel built.

Reproduce after installation:

```bash
python -m pytest -q
python -m pip check
python examples/create_demo_assets.py
asset-spheres --mesh outputs/demo/cup.obj --unit-scale-m 1 \
  --config examples/configs/demo_auto.json --output outputs/cup_spheres_new.json
python -m build
```

The five extracted algorithm source files match the recorded SHA256 values byte
for byte. The original project source files were not modified. Machine-specific
asset paths and provenance-only references were removed from copied presets.

These checks validate extraction, installation and the tested geometric cases.
They do not validate physical robot execution, continuous collision coverage,
all real dataset assets, or every Python/dependency/platform combination.
