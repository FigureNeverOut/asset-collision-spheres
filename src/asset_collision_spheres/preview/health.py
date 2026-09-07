"""Check the Isaac viewport binary before Kit can crash in the dynamic loader.

The optional repair restores ONE library from an explicitly supplied cached
wheel. It checks the installed RECORD hash and matching package version, keeps
a backup, and never upgrades packages or changes preview USD/sphere data.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path


def digest(data):
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def library_record():
    try:
        distribution = importlib.metadata.distribution("isaacsim-extscache-kit")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Isaac package missing: activate the fastsim_vnext environment"
        ) from exc
    records = [
        f
        for f in distribution.files or []
        if str(f).endswith("/pxr/CameraUtil/_cameraUtil.so")
    ]
    if len(records) != 1 or not records[0].hash or records[0].hash.mode != "sha256":
        raise RuntimeError(
            "Cannot uniquely verify Isaac CameraUtil from the installed RECORD"
        )
    record = records[0]
    return distribution, record, Path(distribution.locate_file(record))


def verify_viewport_library():
    distribution, record, path = library_record()
    actual = digest(path.read_bytes()) if path.is_file() else "missing"
    if actual != record.hash.value:
        raise RuntimeError(
            f"Isaac viewport library is damaged or modified: {path}\n"
            f"RECORD SHA256: {record.hash.value}; actual: {actual}\n"
            "Stopped BEFORE starting Kit. This is not a cuRobo sphere-fitting error. "
            "Restore the matching isaacsim-extscache-kit library from a verified wheel; "
            "see docs/1g_book碰撞球静态预览.md."
        )
    return {
        "package_version": distribution.version,
        "library": str(path),
        "sha256": actual,
    }


def restore_from_wheel(wheel_path, backup_root):
    distribution, record, path = library_record()
    before = path.read_bytes()
    if digest(before) == record.hash.value:
        return {"changed": False, **verify_viewport_library()}
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_path = (
            f"isaacsim_extscache_kit-{distribution.version}.dist-info/METADATA"
        )
        if metadata_path not in wheel.namelist():
            raise RuntimeError(
                "Cached wheel does not match the installed package version"
            )
        replacement = wheel.read(str(record))
    if digest(replacement) != record.hash.value:
        raise RuntimeError(
            "Cached library does not match installed RECORD; nothing changed"
        )
    backup_root = Path(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix="isaac_viewport_", dir=backup_root))
    backup = backup_dir / "_cameraUtil.so.before"
    shutil.copy2(path, backup)
    if backup.read_bytes() != before:
        raise RuntimeError("Library changed during backup; refusing repair")
    # Binary package extraction (not a source-code edit). Atomic replacement
    # prevents a partially written shared object from being loaded by Kit.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".cameraUtil-", delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(replacement)
        file.flush()
        os.fsync(file.fileno())
    try:
        temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        "changed": True,
        "backup": str(backup),
        "wheel": str(Path(wheel_path).resolve()),
        "before_sha256": digest(before),
        **verify_viewport_library(),
    }
    (backup_dir / "repair.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-from-wheel", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    if args.restore_from_wheel and not args.backup_dir:
        parser.error("--restore-from-wheel requires --backup-dir")
    result = (
        restore_from_wheel(args.restore_from_wheel, args.backup_dir)
        if args.restore_from_wheel
        else verify_viewport_library()
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
