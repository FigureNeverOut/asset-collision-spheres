"""Create small synthetic assets in ./outputs/demo (no dataset download)."""
from pathlib import Path
import numpy as np
import trimesh


def main():
    output = Path("outputs/demo")
    output.mkdir(parents=True, exist_ok=True)
    profiles = {
        "cup": [[0, 0], [0.04, 0], [0.04, 0.1], [0.035, 0.1], [0.035, 0.008], [0, 0.008]],
        "bowl": [[0, 0], [0.02, 0], [0.05, 0.08], [0.045, 0.08], [0.016, 0.006], [0, 0.006]],
    }
    for name, profile in profiles.items():
        trimesh.creation.revolve(np.array(profile), sections=24).export(output / f"{name}.obj")
    trimesh.creation.box(extents=[0.1, 0.08, 0.04]).export(output / "box.obj")
    print(output)


if __name__ == "__main__":
    main()
