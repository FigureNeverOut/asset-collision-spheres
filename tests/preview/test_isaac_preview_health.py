from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def health(monkeypatch, tmp_path):
    from importlib import import_module
    module = import_module("asset_collision_spheres.preview.health")
    good = b"verified test library"
    record = importlib.metadata.PackagePath(
        "isaacsim/extscache/test/pxr/CameraUtil/_cameraUtil.so"
    )
    record.hash = importlib.metadata.FileHash("sha256=" + module.digest(good))
    path = tmp_path / "_cameraUtil.so"
    path.write_bytes(good)
    distribution = SimpleNamespace(version="6.0.1.0")
    monkeypatch.setattr(module, "library_record", lambda: (distribution, record, path))
    return module, record, path, good


def make_wheel(tmp_path, record, data, version="6.0.1.0"):
    path = tmp_path / "cached-wheel.body"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"isaacsim_extscache_kit-{version}.dist-info/METADATA",
            f"Version: {version}\n",
        )
        wheel.writestr(str(record), data)
    return path


def test_good_library_passes_without_kit(health):
    module, _, path, good = health
    assert module.verify_viewport_library()["sha256"] == module.digest(good)
    assert path.read_bytes() == good


@pytest.mark.parametrize("missing", [False, True])
def test_corrupt_or_missing_library_fails_before_simulation_app(
    health, monkeypatch, missing
):
    _module, _, path, _ = health
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    monkeypatch.setitem(
        sys.modules,
        "isaacsim",
        SimpleNamespace(
            SimulationApp=lambda *a, **kw: pytest.fail("Kit must not start")
        ),
    )
    from asset_collision_spheres.preview.isaac import open_isaac

    with pytest.raises(RuntimeError, match="Stopped BEFORE starting Kit"):
        open_isaac(Path("unused.usda"), SimpleNamespace())


def test_restore_keeps_backup_and_validates_record(health, tmp_path):
    module, record, path, good = health
    path.write_bytes(b"corrupt")
    wheel = make_wheel(tmp_path, record, good)
    report = module.restore_from_wheel(wheel, tmp_path / "backups")
    assert report["changed"]
    assert Path(report["backup"]).read_bytes() == b"corrupt"
    assert path.read_bytes() == good
    assert (
        json.loads((Path(report["backup"]).parent / "repair.json").read_text())
        == report
    )
    assert not module.restore_from_wheel(wheel, tmp_path / "backups")["changed"]


@pytest.mark.parametrize(
    "version,data", [("6.0.0", b"verified test library"), ("6.0.1.0", b"wrong wheel")]
)
def test_mismatched_wheel_cannot_overwrite_installed_library(
    health, tmp_path, version, data
):
    module, record, path, _ = health
    path.write_bytes(b"corrupt")
    wheel = make_wheel(tmp_path, record, data, version)
    with pytest.raises(RuntimeError, match="does not match"):
        module.restore_from_wheel(wheel, tmp_path / "backups")
    assert path.read_bytes() == b"corrupt"
    assert not (tmp_path / "backups").exists()
