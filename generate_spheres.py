"""Source-checkout CLI: python generate_spheres.py --help (no installation)."""

import sys
from pathlib import Path

if __name__ == "__main__":
    # Select THIS checkout even if an older package is installed in the environment.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from asset_collision_spheres.cli import main

    raise SystemExit(main())
