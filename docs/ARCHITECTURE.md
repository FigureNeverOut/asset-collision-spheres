# Source API architecture

This is a source-level interface for colleagues with the existing Python dependencies.
It does not require installing this project or running an HTTP server.

```text
Colleague's program
  → src/asset_collision_spheres/api.py       public exports
  → contracts.py                           typed request/response
  → service.py                             mesh / arrays / file facade
  → algorithms/geometry_policy.py           existing v4 routing
      ├─ convex_surface.py                 existing v3
      └─ original_surface.py               existing v4 graph/grid sampling
  → SphereResult                           metre-based NumPy [N,4] + diagnostics

generate_spheres.py → cli.py                source CLI, no installation
adapters/fastsim.py → existing algorithms   unchanged native integration
```

## Responsibilities

- `contracts.py`: four public settings, result accessors and exclusive JSON saving.
- `service.py`: validate input boundaries, apply explicit file/array units, delegate
  generation. No semantic labels, graph parameters, native slots or workspace lookup.
- `api.py` / lazy `__init__.py`: one documented import namespace. Legacy exports remain.
- `algorithms/`: sole numerical implementation. Do not duplicate algorithms in the facade.
- `loaders/`: file topology, units and authored transforms. USD import is optional/lazy.
- `preview/`: optional diagnostics; core generation never imports Isaac/USD/Torch.
- `adapters/`: optional application integration; no simulator-specific configuration
  is required by a colleague who only wants the local sphere array.

The facade does not infer an object's attachment frame. Input Trimesh is assumed
to be in metres; arrays/files require explicit `unit_scale_m`. USD uses its existing
stage-transformed Z-up output convention, documented on the result.

## Compatibility

Legacy top-level module aliases still resolve to the same modules and classes.
Sampling strategies and thresholds are unchanged. The v4 ray helper also handles
empty hit arrays across Trimesh versions; no-hit distances remain infinity.
The new `generate` default is v4 auto; historical generation functions and configs
retain their defaults. The native adapter can continue using its existing entry points.

Packaging metadata already present in the repository is retained for compatibility,
but no new wheel/sdist/CI publication machinery is introduced for this handoff.
