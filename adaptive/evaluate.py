#!/usr/bin/env python3

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import statistics
import time
import learning as l
import pipeline as p


def evaluate_one(args, meta):
    assert meta['partition'] not in ['train', 'validation'], 'held-out evaluation required'
    jobs = [('conventional', 'conventional', 'none'), ('analytical', 'analytical', 'none'),
            ('nominal_mlp', 'mlp', args.nominal/'shared.txt'), ('shared_mlp', 'mlp', args.models/'shared.txt'),
            ('per_embodiment_mlp', 'mlp', args.models/f"per_{meta['dof']}dof.txt")]
    if meta['dof'] == 4:
        jobs.append(('transfer_3_to_4', 'transfer', args.models/'per_3dof.txt'))
    runs = []
    for name, controller, model in jobs:
        path = args.output/meta['episode']/name
        if path.exists():
            raise RuntimeError('Evaluation exists; preserve it and use a new output directory')
        result = l.run_native(args.exe, args.examples, path, meta, controller, args.bounds, model)
        for key in ['seed', 'family', 'initial_q_rad', 'goal_q_rad', 'physics', 'command_ticks', 'planned_ticks', 'disturbance']:
            if key in meta:
                assert result[key] == meta[key], f'paired condition mismatch {key}'
        runs.append(dict(method=name, episode=meta['episode'], family=meta['family'], dof=meta['dof'],
            partition=meta['partition'], category=meta['category'], severity=meta['severity'], metrics=result['metrics'],
            outcome=result['outcome'], task_success=result['task_success'], fallbacks=result['fallbacks'],
            inference_s=result['inference_s'], completed_ticks=result['completed_ticks'], planned_ticks=result['planned_ticks'],
            wall_s=result['orchestrator_wall_s'], trace_sha256=result['trace_sha256'], model_sha256=result['inference_model_sha256'], path=str(path)))
    return runs


def aggregate(runs):
    result = []
    grouping = [('overall', lambda m: 'all'), ('partition', lambda m: m['partition']),
                ('category', lambda m: m['category']), ('embodiment', lambda m: f"{m['dof']}dof"),
                ('partition_category', lambda m: m['partition']+':'+m['category'])]
    for dimension, keyfn in grouping:
        for group in sorted(set(keyfn(r) for r in runs)):
            group_runs = [r for r in runs if keyfn(r) == group]
            for method in sorted(set(r['method'] for r in group_runs)):
                selected = [r for r in group_runs if r['method'] == method]
                rows = dict(dimension=dimension, group=group, method=method, episodes=len(selected),
                    success_rate=statistics.mean(r['task_success'] for r in selected),
                    unsafe_episodes=sum(r['outcome'] not in ['Success', 'NonConvergence'] for r in selected),
                    fallback_ticks=sum(r['fallbacks'] for r in selected),
                    mean_inference_us_per_joint_tick=sum(r['inference_s'] for r in selected)/max(1,sum(r['completed_ticks']*r['dof'] for r in selected))*1e6,
                    simulated_s=sum(r['completed_ticks']*.002 for r in selected), wall_s=sum(r['wall_s'] for r in selected))
                for metric in ['joint_rmse_rad', 'peak_error_rad', 'command_end_cartesian_m', 'final_cartesian_m', 'final_joint_peak_rad', 'contact_observations']:
                    rows['mean_'+metric] = statistics.mean(r['metrics'][metric] for r in selected)
                    rows['max_'+metric] = max(r['metrics'][metric] for r in selected)
                baseline = {r['episode']: r for r in group_runs if r['method'] == 'conventional'}
                nominal = {r['episode']: r for r in group_runs if r['method'] == 'nominal_mlp'}
                rows['paired_rmse_improvement_vs_conventional_pct'] = statistics.mean(100*(1-r['metrics']['joint_rmse_rad']/baseline[r['episode']]['metrics']['joint_rmse_rad']) for r in selected)
                rows['paired_rmse_improvement_vs_nominal_mlp_pct'] = statistics.mean(100*(1-r['metrics']['joint_rmse_rad']/nominal[r['episode']]['metrics']['joint_rmse_rad']) for r in selected)
                result.append(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['dataset', 'output', 'exe', 'bounds', 'models', 'nominal']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--examples', type=Path, default=p.ROOT/'examples')
    parser.add_argument('--min-index', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    for key in ['dataset', 'output', 'exe', 'bounds', 'models', 'nominal', 'examples']:
        setattr(args, key, getattr(args, key).resolve())
    metas = [json.loads(f.read_text()) for f in sorted(args.dataset.glob('e*/metadata.json'))]
    selected = [m for m in metas if m['partition'] not in ['train', 'validation'] and m['index'] >= args.min_index][:args.episodes]
    assert selected and 1 <= args.workers <= 8

    teacher_report = json.loads((args.models/'training_report.json').read_text())
    train_families = set()
    for model in teacher_report['models']:
        train_families.update(model['fit_families'])
        train_families.update(model['validation_families'])
    assert not train_families & {m['family'] for m in selected}, 'train/eval family leakage: stop for review'
    assert len({m['seed'] for m in selected}) == len(selected), 'duplicate evaluation seeds'
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Evaluation output is not empty; preserve it and use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    p.atomic_json(args.output/'evaluation_config.json', dict(episodes=[m['episode'] for m in selected],
        families=sorted({m['family'] for m in selected}), source_tree_sha256=p.source_hash(), exe_sha256=p.digest(args.exe),
        bounds_sha256=p.digest(args.bounds), training_report_sha256=p.digest(args.models/'training_report.json')))
    began, runs = time.perf_counter(), []
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(evaluate_one, args, m) for m in selected]
        for future in futures.as_completed(pending):
            runs.extend(future.result())
            p.atomic_json(args.output/'runs.json', sorted(runs, key=lambda r: (r['episode'], r['method'])))
            if len(runs) % 20 < 6:
                print(f'{len(runs)} closed-loop controller runs complete', flush=True)
    summary = aggregate(runs)
    p.atomic_json(args.output/'summary.json', dict(wall_s=time.perf_counter()-began, episodes=len(selected), controller_runs=len(runs), groups=summary))
    lines = ['# Measured closed-loop comparison', '', f'{len(selected)} held-out episodes, {len(runs)} controller runs. All metrics measured in MuJoCo.', '',
        '| Group | Method | n | RMSE rad | Command-end m | Final m | Success | Unsafe | RMSE gain vs A |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        if row['dimension'] not in ['overall', 'partition', 'category']:
            continue
        lines.append(f"| {row['group']} | {row['method']} | {row['episodes']} | {row['mean_joint_rmse_rad']:.6f} | {row['mean_command_end_cartesian_m']:.6f} | {row['mean_final_cartesian_m']:.6f} | {row['success_rate']:.1%} | {row['unsafe_episodes']} | {row['paired_rmse_improvement_vs_conventional_pct']:.1f}% |")
    lines += ['', 'Transfer rows cover only 4-DOF episodes; compare paired conditions within the same group.',
              'This dataset assigns severity/trajectory groups to 3-DOF and combination/transfer groups to 4-DOF; shift and embodiment are partially confounded.',
              'Command-phase RMSE excludes fixed settlement; peak error includes settlement. Final error is after a fixed 1 s settlement window.',
              'A success requires joint error <= 0.01 rad, Cartesian error <= 0.005 m and speed <= 0.01 rad/s at the final endpoint.',
              'This endpoint criterion does not claim the production controller\'s 0.25 s dwell criterion.',
              'Sampled path checking is not continuous collision detection. Rejections and failures are retained in run artifacts.']
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    for row in summary:
        if row['dimension'] == 'overall':
            print(json.dumps(row))

if __name__ == '__main__':
    main()
