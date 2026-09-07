"""Locations for bundled configs, local overrides, and optional workspace data."""
from importlib.resources import files
import os
from pathlib import Path


def repository_root():
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir() else None


def config_path(name, *, local=True):
    """Return a bundled preset, optionally overridden by configs/local in a checkout."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Config name must be a relative path within configs")
    root = repository_root()
    override_root = os.environ.get("ASSET_SPHERES_CONFIG_DIR")
    if local:
        directory = Path(override_root).expanduser() if override_root else root / "configs/local" if root else None
        if directory is not None and (directory / relative).is_file():
            return directory / relative
    return Path(str(files("asset_collision_spheres.configs").joinpath(*relative.parts)))


def output_root():
    override = os.environ.get("ASSET_SPHERES_OUTPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (repository_root() or Path.cwd()) / "outputs"


def workspace_root(*, required=True):
    """Find only the explicitly selected workspace or siblings of this checkout."""
    override = os.environ.get("ASSET_SPHERES_WORKSPACE")
    root = repository_root()
    candidate = Path(override).expanduser().resolve() if override else root.parent if root else None
    if candidate and (candidate / "task2sim").is_dir() and (candidate / "FastSim-Demo").is_dir():
        return candidate
    if required:
        raise RuntimeError("This adapter needs a FastSim workspace. Set ASSET_SPHERES_WORKSPACE to its root.")
    return None
