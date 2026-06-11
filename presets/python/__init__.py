"""
Usage:

from presets.python.cantilever import scene
from presets.python.bridge import scene as bridge

grid = scene.to_grid()
scene.save("my_copy.json")
"""

from pathlib import Path

_presets = {}

for _fp in Path(__file__).parent.glob("*.py"):
    _name = _fp.stem
    if _name.startswith("_"):
        continue
    try:
        _mod = __import__(f"presets.python.{_name}", fromlist=["scene"])
        if hasattr(_mod, "scene"):
            _presets[_name] = _mod.scene
    except Exception:
        pass

def list_presets():
    return sorted(_presets)

def get(name):
    return _presets.get(name)

__all__ = ["list_presets", "get"]