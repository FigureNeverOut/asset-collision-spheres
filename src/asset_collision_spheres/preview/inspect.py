"""Open a generated diagnostic USD without fitting, playing physics, or saving edits.

Unlike preview.mesh --open-existing this explicit file viewer does not attest to
the source/config hash; use the accompanying result.json for experiment evidence.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument(
        "--display",
        choices=("overlay", "spheres", "mesh", "features", "convex_spheres", "convex_features", "convex_hull", "convex_sites", "geometry_spheres", "geometry_analysis"),
        default="spheres",
    )
    parser.add_argument(
        "--view", choices=("close", "top", "side", "robot"), default="close"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if not args.usd.is_file():
        parser.error("--usd must be an existing generated diagnostic USD")
    if args.headless and args.frames < 1:
        parser.error("--headless requires positive --frames")
    library = str(Path(sys.prefix) / "lib")
    if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != library:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = library + ":" + env.get("LD_LIBRARY_PATH", "")
        os.execvpe(
            sys.executable,
            [
                sys.executable,
                "-m",
                "asset_collision_spheres.preview.inspect",
                *sys.argv[1:],
            ],
            env,
        )
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    args.camera_path = "/World/Camera" + args.view.title()
    if args.display.startswith("geometry_"):
        args.hidden_paths = ["/World/ReferenceMesh", "/World/ReferenceWire", "/World/Features"]
        if args.display == "geometry_spheres": args.hidden_paths += ["/World/RoutingEvidence"]
        else: args.hidden_paths += ["/World/Spheres", "/World/TargetWire"]
        args.visible_paths = ["/World/RoutingEvidence"] if args.display == "geometry_analysis" else []
    elif args.display.startswith("convex_"):
        from .convex import hidden_paths
        args.hidden_paths = hidden_paths(args.display.removeprefix("convex_"))
    else:
        args.hidden_paths = {
        "overlay": ["/World/Features"],
        "spheres": ["/World/ReferenceMesh", "/World/Features"],
        "mesh": ["/World/Spheres", "/World/Features", "/World/ReferenceWire"],
        "features": ["/World/Spheres", "/World/ReferenceMesh", "/World/ReferenceWire"],
        }[args.display]
    args.preview_legend = "DIAGNOSTIC ONLY | orange=spheres; purple=original material wire; feature view: colors=planes, cyan=edges, red=anchors"
    if args.display.startswith("convex_"):
        args.preview_legend = "CONVEX V3 | cyan=hull; red=corner; cyan=edge; gold=rim; green=face; blue=generic | static diagnostic"
    if args.display.startswith("geometry_"):
        args.preview_legend = "GEOMETRY V4 | cyan=selected hull target; purple=original target | analysis: orange=analysis hull/sections, green=eligible cap, red=rejected"
    from .isaac import open_isaac

    open_isaac(args.usd.resolve(), args)


if __name__ == "__main__":
    main()
