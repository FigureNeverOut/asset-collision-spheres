"""Compatibility entry point; implementation lives in asset_collision_spheres.algorithms.analytic."""
from importlib import import_module
import sys

_impl = import_module("asset_collision_spheres.algorithms.analytic")


def __getattr__(name):
    return getattr(_impl, name)

sys.modules[__name__] = _impl
