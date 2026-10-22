import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=Path)
    p.add_argument('--after', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    changed = git('diff', '--name-only', '94899ed', '--', 'src', 'include', 'examples', 'adaptive', 'interactive/vendor').decode().splitlines()
    assert not changed, changed
    vendor = json.loads((ROOT/'interactive/vendor/manifest.json').read_text())
    for name, expected in vendor['files'].items():
        assert hashlib.sha256((ROOT/'interactive/vendor/low_cost_robot_arm'/name).read_bytes()).hexdigest() == expected, name
    ruckig = ROOT/'experiments/ruckig/vendor'
    blobs = (ruckig/'git-blobs.txt').read_text().splitlines()
    for line in blobs:
        expected, name = line.split(maxsplit=1)
        assert git('hash-object', str(ruckig/name)).decode().strip() == expected, name
    double_s = ROOT/'experiments/double_s/upstream'
    double_files = (double_s/'SHA256SUMS').read_text().splitlines()
    for line in double_files:
        expected, name = line.split(maxsplit=1)
        assert hashlib.sha256((double_s/name).read_bytes()).hexdigest() == expected, name
    names = git('ls-files', '-z').decode().split('\0')
    oversized, forbidden, secrets = [], [], []
    patterns = [rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                rb'\bgh[pousr]_[A-Za-z0-9]{30,}', rb'\bgithub_pat_[A-Za-z0-9_]{30,}',
                rb'\bsk-[A-Za-z0-9_-]{32,}', rb'(?i)bearer\s+[A-Za-z0-9_-]{35,}']
    sizes = []
    for name in filter(None, names):
        path = ROOT/name
        if not path.is_file():
            continue
        size = path.stat().st_size
        sizes.append((size, name))
        if size > 5*1024*1024:
            oversized.append(name)
        if name.startswith(('adaptive/data/', 'adaptive/runs/', 'interactive/runs/', 'integration/runs/', 'build')) or path.suffix.lower() in ('.mp4', '.env', '.pem', '.key') or path.name == '.env':
            forbidden.append(name)
        if size < 5*1024*1024 and path.suffix.lower() not in ('.png', '.jpg', '.stl', '.gz'):
            raw = path.read_bytes()
            for pattern in patterns:
                if re.search(pattern, raw):
                    secrets.append(name)
                    break
    assert not oversized, oversized
    assert not forbidden, forbidden
    assert not secrets, secrets
    report = dict(core_research_assets_unchanged_from='94899ed', menagerie_files_verified=len(vendor['files']),
                  ruckig_files_verified=len(blobs), double_s_files_verified=len(double_files),
                  tracked_files_scanned=len(sizes), oversized_files=oversized, forbidden_artifacts=forbidden,
                  credential_pattern_findings=secrets, largest_files=sorted(sizes, reverse=True)[:10],
                  scan_limit='Known credential patterns and prohibited paths; not a proof that arbitrary secrets cannot exist.')
    if args.before and args.after:
        checks = []
        for before in sorted(args.before.glob('*/comparison.json')):
            after = args.after/before.parent.name/'comparison.json'
            left, right = json.loads(before.read_text()), json.loads(after.read_text())
            assert left['targets'] == right['targets']
            for a, b in zip(left['controllers'], right['controllers'], strict=True):
                for key in ('controller', 'status', 'reason', 'success', 'initial_q', 'goal_q', 'physics'):
                    assert a[key] == b[key], (before, key)
                ma = {k:v for k,v in a['metrics'].items() if k != 'wall_s'}
                mb = {k:v for k,v in b['metrics'].items() if k != 'wall_s'}
                assert ma == mb, (before, a['controller'], ma, mb)
                method = a['controller']
                ta = json.loads((before.parent/method/'execution.json').read_text())['trace']
                tb = json.loads((after.parent/method/'execution.json').read_text())['trace']
                assert ta == tb, (before, method, 'observed trace changed')
                checks.append(dict(case=before.parent.name, controller=method, samples=len(ta), exact_trace_match=True))
        assert len(checks) == 30, len(checks)
        report['closed_loop_comparisons'] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('largest_files','closed_loop_comparisons')}, indent=2))


if __name__ == '__main__':
    main()
