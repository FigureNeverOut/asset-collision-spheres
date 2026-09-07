"""Optional Isaac viewport, initialized before USD/numerical imports."""
def open_isaac(path, args):
    from .health import verify_viewport_library

    print(
        "[preview] Checking Isaac viewport library against package RECORD...",
        flush=True,
    )
    verify_viewport_library()
    print(
        "[preview] Starting Isaac (no sphere fitting for --open-existing)...",
        flush=True,
    )
    # Must initialize Kit before importing omni.*. No World/play/physics is created.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 900})
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport

        print(f"[preview] Opening static USD: {path}", flush=True)
        if not omni.usd.get_context().open_stage(str(path)):
            raise RuntimeError(f"Cannot open {path}")
        if getattr(args, "hidden_paths", None) or getattr(args, "visible_paths", None):
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            # Display-only overrides: never save visibility changes to the USD.
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for prim_path in getattr(args, "hidden_paths", []):
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        UsdGeom.Imageable(prim).MakeInvisible()
                for prim_path in getattr(args, "visible_paths", []):
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        UsdGeom.Imageable(prim).MakeVisible()
        for _ in range(60):
            app.update()
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(
                getattr(args, "camera_path", None)
                or (
                    "/World/CameraRobot"
                    if args.view == "robot"
                    else "/World/CameraClose"
                )
            )
        print(
            getattr(
                args,
                "preview_legend",
                "STATIC PREVIEW READY | orange=baseline, blue=structure, teal=collision mesh, translucent=book, purple=initial outline",
            ),
            flush=True,
        )
        print("[preview] READY: static viewport loaded; no Play required.", flush=True)
        frames = 0
        while app.is_running():
            app.update()
            frames += 1
            if args.frames and frames >= args.frames:
                break
        if args.screenshot and viewport:
            import asyncio
            import time

            from omni.kit.viewport.utility import capture_viewport_to_file

            capture = capture_viewport_to_file(viewport, str(args.screenshot.resolve()))
            task = asyncio.ensure_future(capture.wait_for_result())
            deadline = time.monotonic() + 30
            while not task.done() and time.monotonic() < deadline:
                app.update()
            if not task.done():
                raise RuntimeError("Screenshot timed out")
            task.result()
    finally:
        app.close()
