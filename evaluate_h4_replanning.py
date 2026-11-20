"""Compatibility shim. The implementation is organized under scripts/h4_replanning/."""
from pathlib import Path as _Path
import importlib.util as _importlib_util
import runpy as _runpy
import sys as _sys

_root = _Path(__file__).resolve().parent
_target = _root / "scripts" / "h4_replanning" / "evaluate_h4_replanning.py"
for _folder in (_root, *(_root / "scripts").iterdir()):
    if _folder.is_dir() and str(_folder) not in _sys.path:
        _sys.path.insert(0, str(_folder))
if __name__ == "__main__":
    _runpy.run_path(str(_target), run_name="__main__")
else:
    _spec = _importlib_util.spec_from_file_location(__name__, _target)
    _module = _importlib_util.module_from_spec(_spec)
    _sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update({key: value for key, value in vars(_module).items() if not key.startswith("__")})
