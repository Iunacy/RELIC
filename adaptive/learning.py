#!/usr/bin/env python3

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import concurrent.futures as futures
import pipeline as p


def read_rows(folder):
    with p.open_trace(folder) as f:
        return list(csv.DictReader(f))


def bound_values(path):
    words = path.read_text().split()
    assert words[:2] == ['RELIC_BOUNDS_V1', '4'] and len(words) == 10
    return [float(words[2+2*j]) for j in range(4)], [float(words[3+2*j]) for j in range(4)]


def local_features(m, row, older_error, previous_correction):

    j, n = int(row['joint']), m['dof']
    links = [.30, .25] if n == 3 else [.28, .23]
    local = links[0] if j == 1 else links[1] if j == 2 else (.06 if n == 3 else .07) if j == 3 else 0.
    upstream = links[0] if j == 2 else sum(links) if j == 3 else 0.
    limits = [(-2.8, 2.8), (-1.2, 1.2), (-1.8, 1.8), (-2., 2.)]
    descriptors = m['joint_descriptors'][j*8:(j+1)*8]
    return [float(row[k]) for k in ['q_des_rad', 'v_des_analytic_rad_s', 'a_des_analytic_rad_s2', 'q_pre_rad', 'v_pre_rad_s', 'previous_error_rad']] + [older_error, previous_correction] + descriptors


def examples_from_trace(folder, stride=4, targets=None):
    m = json.loads((folder/'metadata.json').read_text())
    older, previous = [0.]*4, [0.]*4
    for row in read_rows(folder):
        j, tick = int(row['joint']), int(row['tick'])
        if tick:
            if tick % stride == 0:
                x = local_features(m, row, older[j], previous[j])
                yield x, (targets[tick][j] if targets is not None else float(row['correction_rad'])), j
            older[j] = float(row['previous_error_rad'])
            previous[j] = float(row['correction_rad'])


def run_native(exe, examples, out, m, controller, bounds, model='none', control=None):
    out.mkdir(parents=True, exist_ok=False)
    args = [str(exe), str(examples), str(out)] + [str(m[k]) for k in ['seed', 'family', 'dof', 'category', 'severity', 'partition', 'episode']] + [controller, str(bounds), str(model)]
    if control is not None:
        args.append(str(control))
    began = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode:
        p.atomic_json(out/'process_failure.json', dict(returncode=result.returncode, stderr=result.stderr, args=args))
        raise RuntimeError(f'native process failed: {result.stderr}')
    meta = json.loads((out/'native.json').read_text())
    meta.update(index=m['index'], executable_sha256=p.digest(exe), source_tree_sha256=p.source_hash(), source_dirty=True)
    with open(out/'telemetry.csv', 'rb') as inp, open(out/'telemetry.csv.gz', 'wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as gz:
            shutil.copyfileobj(inp, gz)
    (out/'telemetry.csv').unlink()
    meta.update(trace_sha256=p.digest(out/'telemetry.csv.gz'), model_sha256=p.digest(out/'nominal.xml'), orchestrator_wall_s=time.perf_counter()-began,
                bounds_sha256=p.digest(bounds), inference_model_sha256=p.digest(Path(model)) if Path(model).is_file() else None)
    meta['randomized_model_sha256'] = hashlib.sha256((meta['model_sha256'] + json.dumps(meta['physics'], sort_keys=True, separators=(',', ':'))).encode()).hexdigest()
    if control is not None:
        meta['control_configuration'] = Path(control).read_text()
        meta['control_configuration_sha256'] = p.digest(Path(control))
    meta['native_argv'] = args
    p.atomic_json(out/'metadata.json', meta)
    (out/'native.json').unlink()
    p.validate_episode(out)
    return meta


def table_from_trace(folder):
    m = json.loads((folder/'metadata.json').read_text())
    n = m['dof']
    seq = [[0.]*n for _ in range(m['planned_ticks']+1)]
    err = [[0.]*n for _ in seq]
    for row in read_rows(folder):
        k, j = int(row['tick']), int(row['joint'])
        seq[k][j], err[k][j] = float(row['correction_rad']), float(row['error_rad'])
    return seq, err


def proposal(seq, error, bounds, gain):
    magnitude, rate = bound_values(bounds)
    out = [[0.]*len(seq[0])]
    for k in range(1, len(seq)):
        row = []
        for j in range(len(seq[0])):

            neighbors = range(max(1, k-2), min(len(seq), k+3))
            smooth = sum(error[i][j] for i in neighbors)/len(neighbors)
            target = max(-magnitude[j], min(magnitude[j], seq[k][j]+gain*smooth))
            row.append(max(out[-1][j]-rate[j], min(out[-1][j]+rate[j], target)))
        out.append(row)
    return out


def write_table(path, rows):
    with open(path, 'w') as f:
        f.write(f'RELIC_REPLAY_V1 {len(rows[0])} {len(rows)}\n')
        for row in rows:
            f.write(' '.join(format(v, '.17g') for v in row)+'\n')


def acceptance(trial, best):
    safe = trial['outcome'] in ['Success', 'NonConvergence'] and trial['completed_ticks'] == trial['planned_ticks'] and trial['fallbacks'] == 0 and trial['metrics']['contact_observations'] == 0
    improved = trial['metrics']['joint_rmse_rad'] < best['metrics']['joint_rmse_rad']*.995
    final_ok = trial['metrics']['final_cartesian_m'] <= max(best['metrics']['final_cartesian_m']+.00025, .0005)
    passed = safe and improved and final_ok
    why = 'accepted' if passed else 'unsafe_or_incomplete' if not safe else 'no_rmse_improvement' if not improved else 'final_error_regressed'
    return passed, why


def teacher_one(args, m):
    assert m['partition'] == 'train', 'ILC may never use held-out families'
    folder = args.output/'teachers'/m['episode']
    if folder.exists():
        raise RuntimeError('Teacher output already exists; preserve it and use a new output directory')
    baseline = run_native(args.exe, args.examples, folder/'baseline', m, 'conventional', args.bounds)
    best, best_path = baseline, folder/'baseline'
    accepted, records = [], []
    for attempt in range(args.attempts):
        seq, error = table_from_trace(best_path)
        rows = proposal(seq, error, args.bounds, 1.0 if attempt == 0 else .7)
        replay = folder/f'proposal_{attempt}.txt'
        write_table(replay, rows)
        trial_path = folder/f'attempt_{attempt}'
        trial = run_native(args.exe, args.examples, trial_path, m, 'replay', args.bounds, replay)
        passed, why = acceptance(trial, best)
        repeat_hash = None
        if passed:
            repeat = run_native(args.exe, args.examples, folder/f'repeat_{attempt}', m, 'replay', args.bounds, replay)
            repeat_hash = repeat['trace_sha256']
            if repeat_hash != trial['trace_sha256']:
                raise RuntimeError('ILC replay not reproducible; labels invalid, stop for architectural review')
            best, best_path = trial, trial_path
            accepted.append(str(trial_path.resolve()))
        records.append(dict(attempt=attempt, kind='previous_error_replay' if attempt == 0 else 'iterative_error_update', accepted=passed,
                            reason=why, category=m['category'], severity=m['severity'], baseline_rmse=baseline['metrics']['joint_rmse_rad'],
                            trial_rmse=trial['metrics']['joint_rmse_rad'], trial_final_m=trial['metrics']['final_cartesian_m'],
                            outcome=trial['outcome'], fallbacks=trial['fallbacks'], trace_sha256=trial['trace_sha256'], repeat_sha256=repeat_hash,
                            proposal_sha256=p.digest(replay), path=str(trial_path.resolve())))
        p.atomic_json(folder/'attempts.json', records)
    return dict(episode=m['episode'], family=m['family'], dof=m['dof'], partition=m['partition'], category=m['category'], severity=m['severity'],
                accepted=accepted, attempts=records, baseline_rmse=baseline['metrics']['joint_rmse_rad'], best_rmse=best['metrics']['joint_rmse_rad'])


def teachers(args):
    assert 1 <= args.workers <= 8 and 1 <= args.attempts <= 8
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Teacher output is not empty; preserve it and use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    models = [json.loads(f.read_text()) for f in sorted(args.dataset.glob('e*/metadata.json'))]
    selected = [m for m in models if m['partition'] == 'train'][:args.episodes]
    assert selected and {m['dof'] for m in selected} == {3, 4}
    p.atomic_json(args.output/'teacher_config.json', dict(exe_sha256=p.digest(args.exe), bounds_sha256=p.digest(args.bounds), source_tree_sha256=p.source_hash(),
        episodes=[m['episode'] for m in selected], families=sorted({m['family'] for m in selected}), attempts=args.attempts))
    began = time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(teacher_one, args, m) for m in selected]
        records = []
        for f in futures.as_completed(pending):
            records.append(f.result())
            if len(records) % 20 == 0:
                print(f'{len(records)}/{len(selected)} ILC training episodes complete', flush=True)
            p.atomic_json(args.output/'teacher_manifest.json', sorted(records, key=lambda m: m['episode']))
    attempts = [a for m in records for a in m['attempts']]
    summary = dict(training_episodes=len(records), attempts=len(attempts), accepted=sum(a['accepted'] for a in attempts), wall_s=time.perf_counter()-began,
                   rejected_by_category={}, rejected_by_severity={})
    for field, keyfn in [('rejected_by_category', lambda a: a['category']), ('rejected_by_severity', lambda a: 'low_[.25,.45)' if a['severity'] < .45 else 'high_[.45,.68)')]:
        for a in attempts:
            entry = summary[field].setdefault(keyfn(a), dict(attempts=0, rejected=0))
            entry['attempts'] += 1
            entry['rejected'] += not a['accepted']
        for entry in summary[field].values():
            entry['rejection_rate'] = entry['rejected']/entry['attempts']
    p.atomic_json(args.output/'teacher_summary.json', summary)
    print(json.dumps(summary, indent=2))


def train(args):
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.teachers/'teacher_manifest.json').read_text())
    if args.categories != 'all':
        allowed = set(args.categories.split(','))
        manifest = [m for m in manifest if m['category'] in allowed]
    assert all(m['partition'] == 'train' for m in manifest), 'label leakage'
    magnitude, rate = bound_values(args.bounds)
    inputs, labels, joints, families, dofs = [], [], [], [], []
    for m in manifest:
        if not m['accepted']:
            continue



        best_meta = p.validate_episode(Path(m['accepted'][-1]))
        assert best_meta['partition'] == 'train' and best_meta['family'] == m['family'], 'teacher label partition mismatch: stop for review'
        assert any(a['accepted'] and a['path'] == m['accepted'][-1] and a['trace_sha256'] == best_meta['trace_sha256'] and a['repeat_sha256'] == best_meta['trace_sha256'] for a in m['attempts']), 'unvalidated teacher labels: stop for review'
        best_sequence, _ = table_from_trace(Path(m['accepted'][-1]))
        folders = [str(Path(m['accepted'][0]).parent/'baseline')] + m['accepted']
        for folder in folders:
            source_meta = json.loads((Path(folder)/'metadata.json').read_text())
            assert source_meta['partition'] == 'train' and source_meta['family'] == m['family'], 'feature partition mismatch: stop for review'
            targets = best_sequence if args.label_mode == 'best-sequence' else None
            if args.label_mode == 'executed' and Path(folder).name == 'baseline':
                continue
            for x, y, j in examples_from_trace(Path(folder), stride=4, targets=targets):
                inputs.append(x)
                labels.append(y)
                joints.append(j)
                families.append(m['family'])
                dofs.append(m['dof'])
    assert inputs, 'no replay-validated labels; stop for architectural review'
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Training output is not empty; preserve it and use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    j = np.asarray(joints)
    family = np.asarray(families, dtype=np.int64)
    dof = np.asarray(dofs)
    del inputs, labels, joints, families, dofs

    np.savez_compressed(args.output/'prepared.npz', features=x, correction_rad=y, joint=j, family=family, dof=dof)
    records = []
    for embodiment in [0, 3, 4]:
        selected = np.ones(len(x), bool) if embodiment == 0 else dof == embodiment
        unique = sorted(set(family[selected].tolist()))
        assert len(unique) >= 4, 'need independent families per embodiment'

        validation_families = set(unique[::5])
        valid = selected & np.isin(family, list(validation_families))
        fit = selected & ~valid
        mean, scale = x[fit].mean(0), np.maximum(x[fit].std(0), .01)
        xx = torch.from_numpy((x-mean)/scale)
        yy = torch.from_numpy(y/np.asarray(magnitude, np.float32)[j, None])
        model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1), nn.Tanh())
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        train_ids = torch.from_numpy(np.flatnonzero(fit))
        valid_ids = torch.from_numpy(np.flatnonzero(valid))
        best_loss, best_state = math.inf, None
        started = time.perf_counter()
        history = []
        generator = torch.Generator().manual_seed(args.seed+embodiment)
        for epoch in range(args.epochs):
            model.train()
            order = train_ids[torch.randperm(len(train_ids), generator=generator)]
            for batch in order.split(2048):
                optimizer.zero_grad()
                loss = ((model(xx[batch])-yy[batch])**2).mean()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(((model(xx[valid_ids])-yy[valid_ids])**2).mean())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        name = 'shared' if embodiment == 0 else f'per_{embodiment}dof'

        chunks = [magnitude, rate, mean.tolist(), scale.tolist()]
        for layer in [model[0], model[2], model[4]]:
            chunks.extend([layer.weight.detach().numpy().reshape(-1).tolist(), layer.bias.detach().numpy().reshape(-1).tolist()])
        text = f'RELIC_MLP_V1 relic.local.v1 16 32 32 {embodiment}\n' + '\n'.join(' '.join(format(float(v), '.17g') for v in values) for values in chunks)+'\n'
        path = args.output/f'{name}.txt'
        path.write_text(text)
        record = dict(model=name, embodiment=embodiment, fit_rows=int(fit.sum()), validation_rows=int(valid.sum()),
                      fit_families=sorted(set(family[fit].tolist())), validation_families=sorted(validation_families),
                      normalized_validation_mse=best_loss, validation_curve=history, train_s=time.perf_counter()-started,
                      model_sha256=p.digest(path), seed=args.seed, cpu_threads=args.threads, epochs=args.epochs)
        assert not set(record['fit_families']) & validation_families
        p.atomic_json(args.output/f'{name}.json', record)
        records.append(record)
        print(f"{name}: {record['train_s']:.2f}s, validation MSE={best_loss:.6f}", flush=True)
    p.atomic_json(args.output/'training_report.json', dict(models=records, teacher_manifest_sha256=p.digest(args.teachers/'teacher_manifest.json'), bounds_sha256=p.digest(args.bounds), prepared_sha256=p.digest(args.output/'prepared.npz'), feature_version='relic.local.v1', label_mode=args.label_mode, categories=args.categories, training_episodes=len(manifest)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('teachers')
    t.add_argument('--dataset', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True)
    t.add_argument('--exe', type=Path, required=True)
    t.add_argument('--examples', type=Path, default=p.ROOT/'examples')
    t.add_argument('--bounds', type=Path, required=True)
    t.add_argument('--episodes', type=int, default=160)
    t.add_argument('--attempts', type=int, default=3)
    t.add_argument('--workers', type=int, default=3)
    tr = sub.add_parser('train')
    tr.add_argument('--teachers', type=Path, required=True)
    tr.add_argument('--output', type=Path, required=True)
    tr.add_argument('--bounds', type=Path, required=True)
    tr.add_argument('--label-mode', choices=['best-sequence', 'executed'], default='best-sequence')
    tr.add_argument('--categories', default='all')
    tr.add_argument('--epochs', type=int, default=40)
    tr.add_argument('--threads', type=int, default=2)
    tr.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    for key in ['exe', 'examples', 'bounds', 'dataset', 'output', 'teachers']:
        if hasattr(args, key):
            setattr(args, key, getattr(args, key).resolve())
    if args.command == 'teachers':
        teachers(args)
    else:
        train(args)

if __name__ == '__main__':
    main()
