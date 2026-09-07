"""Freeze prior v3 data and unchanged numerical implementations for v4 regression."""

import hashlib
import json
from pathlib import Path

from asset_collision_spheres.paths import output_root


def freeze():
    repo = Path(__file__).resolve().parents[1]
    destination = output_root() / "geometry_v4/frozen_manifest.json"
    if destination.exists():
        hashes = json.loads(destination.read_text())
        changed = [
            p
            for p, sha in hashes.items()
            if not Path(p).is_file()
            or hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha
        ]
        if changed:
            raise RuntimeError(f"Previous results changed: {changed}")
    else:
        paths = [
            p
            for root in [
                repo / "configs",
                output_root() / "convex_v3",
                output_root() / "structure_v2",
            ]
            for p in root.rglob("*")
            if p.is_file()
        ]
        paths += [
            repo / "src/asset_collision_spheres/algorithms" / name
            for name in [
                "convex_surface.py",
                "hull_structure.py",
                "hull_diagnostics.py",
                "joint_selection.py",
                "mesh_regions.py",
            ]
        ]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Verified {len(hashes)} pre-v4 files", flush=True)


if __name__ == "__main__":
    freeze()
