#!/usr/bin/env python3

import argparse
from collections import Counter
import concurrent.futures as futures
import json
from pathlib import Path
import tempfile
import time
import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.dataset/'campaign.json').read_text())
    assert p.digest(args.exe) == identity['executable_sha256'], 'pinned executable mismatch'
    began = time.perf_counter()
    manifest = p.validate_dataset(args.dataset, full=False)
    dofs, outcomes, ranges = Counter(), Counter(), {}
    for folder in sorted(args.dataset.glob('e[0-9]*')):
        m = json.loads((folder/'metadata.json').read_text())
        assert p.digest(folder/'telemetry.csv.gz') == m['trace_sha256'], 'committed trace changed'
        assert p.digest(folder/'nominal.xml') == m['model_sha256'], 'committed nominal model changed'
        dofs[m['dof']] += 1
        outcomes[m['outcome']] += 1
        for k in ['gain_scale', 'force_scale', 'damping_scale', 'friction_Nm', 'mass_scale', 'payload_kg']:
            value = m['physics'][k]
            bounds = ranges.setdefault(k, [value, value])
            bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
    indices = [0, 1, 6, 8, 14, 338, 339, 340, 341, 342, 5006, 9999]
    count = manifest['episodes']
    indices = [i for i in indices if i < count]
    repeats = []
    with tempfile.TemporaryDirectory(prefix='relic-audit-') as temp:
        with futures.ThreadPoolExecutor(max_workers=3) as pool:
            pending = {pool.submit(p.generate_one, args.exe.resolve(), p.ROOT/'examples', Path(temp), p.spec(i, identity['config']), identity): i for i in indices}
            for future in futures.as_completed(pending):
                future.result()
                i = pending[future]
                name = p.spec(i, identity['config'])['episode']
                original = json.loads((args.dataset/name/'metadata.json').read_text())
                replay = json.loads((Path(temp)/name/'metadata.json').read_text())
                assert original['trace_sha256'] == replay['trace_sha256'], 'deterministic replay failed'
                assert original['metrics'] == replay['metrics'] and original['outcome'] == replay['outcome'], 'replay metric mismatch'
                repeats.append(dict(episode=name, byte_identical_trace=True, identical_metrics=True))
    report = dict(episodes=manifest['episodes'], complete_checksum_audit=True, full_row_validation='performed for every episode before generation commit',
        dofs=dict(dofs), outcomes=dict(outcomes), physical_ranges=ranges, deterministic_replays=sorted(repeats, key=lambda r: r['episode']),
        wall_s=time.perf_counter()-began)
    p.atomic_json(args.output, report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
