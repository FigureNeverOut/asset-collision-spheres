"""Attest historical configs and generated data without modifying their bytes."""

import hashlib
import json
from pathlib import Path

from asset_collision_spheres.paths import output_root, workspace_root


def freeze():
    root = workspace_root()
    repo = Path(__file__).resolve().parents[1]
    destination = output_root() / "convex_v3" / "frozen_manifest.json"
    roots = [repo / "configs", output_root() / "structure_v2"]
    roots += [root / "task2sim/runs/1e", root / "task2sim/runs/collision_ball_test"]
    hashes = {}
    for directory in roots:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".json",
                ".usda",
                ".usd",
                ".png",
            }:
                if "convex" not in path.name:
                    hashes[str(path.resolve())] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    if destination.exists():
        previous = json.loads(destination.read_text())
        changed = [p for p, sha in previous.items() if hashes.get(p) != sha]
        if changed:
            raise RuntimeError(f"Historical artifacts changed: {changed}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Verified {len(hashes)} historical config/artifact hashes", flush=True)


if __name__ == "__main__":
    freeze()
