"""Read-only local capability audit. This is never a simulation qualification."""
import argparse
import importlib.util
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

SIM = '5.1.0'
LAB_COMMIT = '37ddf626871758333d6ed89cf64ad702aef127d0'


def command(args):
    try:
        p = subprocess.run(args,capture_output=True,text=True,timeout=15)
        return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'error':str(e)}


def audit():
    packages = {name:importlib.util.find_spec(name) is not None for name in ('isaacsim','isaaclab','torch','numpy','mujoco')}
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'qualification':'BLOCKED','simulation_executed':False,
            'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'os_release':Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else '',
            'gpu':command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
            'memory':command(['free','-b']),'packages':packages,'package_versions':versions,
            'selected_runtime':{'os':'Ubuntu 22.04 x86_64','python':'3.11','isaacsim':SIM,
                                'isaaclab_tag':'v2.3.2','isaaclab_commit':LAB_COMMIT},
            'runtime_tested':False,
            'recovery':'bash embodiment/qualify.sh --headless --device cuda:0 --enable_cameras',
            'blockers':[f'{p} not installed in this interpreter' for p in ('isaacsim','isaaclab') if not packages[p]]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    data = audit()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
