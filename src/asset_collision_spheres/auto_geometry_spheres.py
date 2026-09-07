"""Compatibility entry point; implementation lives in asset_collision_spheres.algorithms.auto_geometry."""
from importlib import import_module
import sys

_impl = import_module("asset_collision_spheres.algorithms.auto_geometry")


def __getattr__(name):
    return getattr(_impl, name)

sys.modules[__name__] = _impl
