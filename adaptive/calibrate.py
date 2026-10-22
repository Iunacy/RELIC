#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from array import array
import pipeline as p


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values)-1, int(q * (len(values)-1)))] if values else 0.


def calibrate(dataset, output):
    error = [array('d') for _ in range(4)]
    rate = [array('d') for _ in range(4)]
    coverage, episodes, source = {}, 0, []
    for folder in sorted(dataset.glob('e[0-9]*')):
        m = json.loads((folder/'metadata.json').read_text())
        if m['partition'] != 'train':
            continue
        episodes += 1
        source.append(m['trace_sha256'])
        coverage[m['category']] = coverage.get(m['category'], 0) + 1
        previous = [0.] * 4
        with p.open_trace(folder) as f:
            for row in csv.DictReader(f):
                j = int(row['joint'])
                e = float(row['error_rad'])
                error[j].append(abs(e))
                rate[j].append(abs(e-previous[j]))
                previous[j] = e
    assert episodes and all(error), 'both embodiments required for named-joint calibration'
    details = []
    for j, name in enumerate(['yaw', 'shoulder', 'elbow', 'wrist']):
        e99, r99 = quantile(error[j], .99), quantile(rate[j], .99)
        details.append(dict(joint=name, samples=len(error[j]), abs_error_p50_rad=quantile(error[j], .5),
            abs_error_p99_rad=e99, peak_error_rad=max(error[j]), error_delta_p99_rad=r99,
            magnitude_rad=min(.16, max(.01, 1.25*e99)), rate_rad_per_tick=min(.0025, max(.0001, 1.5*r99))))
    report = dict(version='relic.bounds.v1', training_episodes=episodes, category_coverage=coverage,
                  calibration_trace_hashes=source, rule='magnitude=clip(1.25*p99_abs_error,.01,.16); rate=clip(1.5*p99_abs_error_delta,.0001,.0025)',
                  timestep_s=.002, joints=details)
    output.mkdir(parents=True, exist_ok=True)
    p.atomic_json(output/'calibration.json', report)
    (output/'bounds.txt').write_text('RELIC_BOUNDS_V1 4\n' + '\n'.join(f"{j['magnitude_rad']:.17g} {j['rate_rad_per_tick']:.17g}" for j in details) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'calibration_trace_hashes'}, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    calibrate(args.dataset, args.output)
