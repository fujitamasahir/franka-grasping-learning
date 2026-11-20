"""Run a role-organized project script from the repository root."""
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
script_root=root/'scripts'
for directory in (root, *(p for p in script_root.iterdir() if p.is_dir())):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
if len(sys.argv)<2:
    raise SystemExit('Usage: python scripts/run.py <group/script.py> [arguments...]')
target=(script_root/sys.argv[1]).resolve()
if not target.is_relative_to(script_root) or not target.is_file():
    raise SystemExit(f'Unknown script: {sys.argv[1]}')
sys.argv=sys.argv[1:]
sys.argv[0]=str(target)
runpy.run_path(str(target),run_name='__main__')
