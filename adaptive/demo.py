#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import learning as l
import pipeline as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--bounds', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    for key in ['exe', 'bounds', 'model', 'output']:
        setattr(args, key, getattr(args, key).resolve())
    cfg = json.loads((p.ROOT/'adaptive/campaign.json').read_text())
    cfg['seed'] = args.seed

    nominal = p.spec(8, cfg)
    degraded = p.spec(14, cfg)
    assert nominal['family'] == degraded['family'] and nominal['partition'] != 'train'
    jobs = [('nominal', nominal, 'conventional'), ('degraded', degraded, 'conventional'),
            ('analytical', degraded, 'analytical'), ('adapted', degraded, 'mlp')]
    measured = []
    traces = {}
    for name, meta, controller in jobs:
        folder = args.output/name
        result = l.run_native(args.exe, p.ROOT/'examples', folder, meta, controller, args.bounds, args.model if controller == 'mlp' else 'none')
        traces[name] = l.read_rows(folder)

        moving = max(abs(float(r['q_actual_rad'])-result['initial_q_rad'][int(r['joint'])]) for r in traces[name])
        assert moving > .01 and result['completed_ticks'] > result['command_ticks'], 'demonstration did not execute motion'
        measured.append(dict(case=name, outcome=result['outcome'], success=result['task_success'], measured_motion_rad=moving, **result['metrics']))
    p.atomic_json(args.output/'feedback.json', measured)
    print('case         RMSE(rad)   command-end(m) final(m)   success')
    for r in measured:
        print(f"{r['case']:12s} {r['joint_rmse_rad']:.6f}    {r['command_end_cartesian_m']:.6f}       {r['final_cartesian_m']:.6f}   {r['success']}")
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        colors = dict(nominal='#666666', degraded='#cc4444', analytical='#d09a1b', adapted='#168578')
        for j, joint in enumerate(['yaw', 'shoulder', 'elbow', 'wrist']):
            for name in ['degraded', 'adapted']:
                rows = [r for r in traces[name] if int(r['joint']) == j]
                times = [float(r['sim_time_s']) for r in rows]
                axes[j, 0].plot(times, [float(r['q_actual_rad']) for r in rows], label=name, color=colors[name])
                if name == 'degraded':
                    axes[j, 0].plot(times, [float(r['q_des_rad']) for r in rows], label='desired', color='black', linestyle='--', linewidth=1)
            for name in traces:
                rows = [r for r in traces[name] if int(r['joint']) == j]
                axes[j, 1].plot([float(r['sim_time_s']) for r in rows], [float(r['error_rad']) for r in rows], label=name, color=colors[name])
            axes[j, 0].set_ylabel(joint+' position (rad)')
            axes[j, 1].set_ylabel('tracking error (rad)')
            for a in axes[j]:
                a.grid(alpha=.2)
        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[-1, 0].set_xlabel('simulation time (s)')
        axes[-1, 1].set_xlabel('simulation time (s)')
        fig.suptitle('RELIC: measured closed-loop MuJoCo motion under combined degradation')
        fig.tight_layout()
        fig.savefig(args.output/'feedback.png', dpi=150)
        plt.close(fig)

if __name__ == '__main__':
    main()
