"""Read a USD preset; relative assets resolve against the preset file."""
import json
from pathlib import Path


def read_preset(path, budget=None):
    path = Path(path).expanduser().resolve()
    preset = json.loads(path.read_text())
    source = Path(preset["source"]["usd_path"]).expanduser()
    if not source.is_absolute():
        source = path.parent / source
    preset["source"]["usd_path"] = str(source.resolve())
    config = dict(preset["generation"])
    if budget is not None:
        config["sphere_budget"] = budget
    return preset, config
