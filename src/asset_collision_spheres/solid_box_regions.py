"""Compatibility entry point; implementation lives in asset_collision_spheres.algorithms.solid_box."""
from importlib import import_module
import sys

_impl = import_module("asset_collision_spheres.algorithms.solid_box")


def __getattr__(name):
    return getattr(_impl, name)

sys.modules[__name__] = _impl
