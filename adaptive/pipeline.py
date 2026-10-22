#!/usr/bin/env python3

import argparse
import concurrent.futures as futures
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

if not __debug__:
    raise RuntimeError('Dataset validation requires Python assertions enabled; do not use -O')

SCHEMA = 'relic.episode.v1'
CATEGORIES = ['nominal', 'gain', 'force', 'friction', 'mass', 'payload', 'combined', 'recovery']
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def spec(index, campaign):
    family, category_index = divmod(index, len(campaign['categories']))
    dof = campaign['embodiments'][family % len(campaign['embodiments'])]
    group = family % 20
    partition = {0: 'trajectory_test', 1: 'combination_test', 2: 'severity_test', 3: 'transfer_test', 4: 'validation'}.get(group, 'train')

    seed = campaign['seed'] * 1000003 + index
    severity = (0.85 + (index % 13) / 100) if partition == 'severity_test' else (0.25 + (index % 43) / 100)
    category = campaign['categories'][category_index]

    if category == 'combined' and partition == 'combination_test':
        severity = 0.78
    return dict(index=index, family=campaign['seed'] * 100003 + family, seed=seed, dof=dof,
                category=category, severity=severity, partition=partition, episode=f'e{index:07d}')


def open_trace(folder):
    p = folder / 'telemetry.csv.gz'
    return gzip.open(p, 'rt', newline='') if p.exists() else open(folder / 'telemetry.csv', newline='')


def validate_episode(folder, check_hash=True):
    m = json.loads((folder / 'metadata.json').read_text())
    assert m['schema_version'] == SCHEMA, 'schema version'
    assert m['dof'] in (3, 4) and m['embodiment'] == f"synthetic_{m['dof']}dof_v1", 'embodiment'
    if check_hash:
        assert digest(folder / 'telemetry.csv.gz') == m['trace_sha256'], 'trace checksum'
        assert digest(folder / 'nominal.xml') == m['model_sha256'], 'model checksum'
        if 'randomized_model_sha256' in m:
            assert hashlib.sha256((m['model_sha256'] + json.dumps(m['physics'], sort_keys=True, separators=(',', ':'))).encode()).hexdigest() == m['randomized_model_sha256'], 'randomized model checksum'
    assert len(m['initial_q_rad']) == len(m['goal_q_rad']) == m['dof'], 'metadata dimensions'
    assert m['completed_ticks'] <= m['planned_ticks'] and m['command_ticks'] <= m['planned_ticks'], 'termination ticks'
    assert m['outcome'] in ['Success', 'NonConvergence', 'InvalidPath', 'DisallowedContact', 'ObservedJointLimit', 'NumericalWarning', 'SimulationError'], 'outcome'
    assert bool(m['termination_reason']) and m['failure_class'] == ('none' if m['task_success'] else m['outcome']), 'failure recording'
    assert m['task_success'] == (m['outcome'] == 'Success'), 'outcome/success consistency'
    if m['task_success']:
        assert m['completed_ticks'] == m['planned_ticks'] and m['metrics']['final_joint_peak_rad'] <= .01 and m['metrics']['final_cartesian_m'] <= .005 and m['metrics']['final_velocity_peak_rad_s'] <= .01, 'success tolerances'
    p = m['physics']
    for key in ['gain_scale', 'force_scale', 'damping_scale', 'mass_scale']:
        assert 0.1 <= p[key] <= 4 and math.isfinite(p[key]), 'physical scales'
    assert 0 <= p['payload_kg'] <= 0.5 and 0 <= p['friction_Nm'] <= 2, 'payload/friction'
    assert len(p['actuator_kp']) == len(p['joint_damping_Nm_s_rad']) == m['dof'], 'physics dimensions'
    assert all(x > 0 for x in p['body_mass_kg'][1:] if x != 0), 'mass positivity'
    inertias = p['body_inertia_kg_m2']
    for k in range(0, len(inertias), 3):
        vals = inertias[k:k+3]
        assert min(vals) >= 0 and 2 * max(vals) <= sum(vals) + 1e-10, 'physical inertia triangle'
    count, squared, peak, n_command = 0, 0., 0., 0
    prior = [None] * m['dof']
    names = ['yaw', 'shoulder', 'elbow', 'wrist']
    text_fields = {'episode', 'stage', 'phase', 'embodiment', 'joint_name', 'safety_status', 'fallback_reason'}
    with open_trace(folder) as stream:
        for row in csv.DictReader(stream):
            x = {k: float(v) for k, v in row.items() if k not in text_fields}
            assert all(math.isfinite(v) for v in x.values()), 'nonfinite telemetry'
            tick, joint = divmod(count, m['dof'])
            assert x['tick'] == tick and x['joint'] == joint and row['joint_name'] == names[joint], 'missing/duplicate/mapped samples'
            assert row['episode'] == m['episode'] and row['embodiment'] == m['embodiment'], 'row metadata'
            assert abs(x['sim_time_s'] - tick * m['timestep_s']) < 1e-9 and abs(x['sim_time_s'] - x['reference_time_s']) < 1e-9, 'time ordering'
            assert row['stage'] == ('initial' if tick == 0 else 'post'), 'stage'
            assert row['phase'] == ('initial' if tick == 0 else 'command' if tick <= m['command_ticks'] else 'settling'), 'phase'
            assert abs(x['error_rad'] - (x['q_des_rad'] - x['q_actual_rad'])) < 1e-12, 'tracking error'
            assert abs(x['ctrl_rad'] - x['q_des_rad'] - x['correction_rad']) < 1e-12, 'command/correction'
            assert abs(x['correction_rad']) <= m.get('correction_magnitude_rad', [0, 0, 0, 0])[joint] + 1e-12, 'residual bound'
            assert x['ctrl_lower_rad'] <= x['ctrl_rad'] <= x['ctrl_upper_rad'], 'control limits'
            assert x['ctrl_lower_rad'] <= x['q_actual_rad'] <= x['ctrl_upper_rad'] or row['safety_status'] == 'ObservedJointLimit', 'observed joint limits'
            for key in ['actuator_force_Nm', 'interval_start_force_Nm']:
                assert x['force_lower_Nm'] - 1e-8 <= x[key] <= x['force_upper_Nm'] + 1e-8, 'actuator limits'
            expected_force = max(x['force_lower_Nm'], min(x['force_upper_Nm'], x['servo_unclipped_Nm']))
            assert abs(x['actuator_force_Nm'] - expected_force) < 1e-8 and abs(x['qfrc_actuator_Nm'] - expected_force) < 1e-8, 'effort mapping'
            assert abs(x['v_des_analytic_rad_s']) <= m['velocity_limit_rad_s'] + 1e-8 and abs(x['a_des_analytic_rad_s2']) <= m['acceleration_limit_rad_s2'] + 1e-8, 'reference derivative limits'
            if tick == 0:
                assert x['q_actual_rad'] == m['initial_q_rad'][joint] and x['v_actual_rad_s'] == 0 and x['previous_error_rad'] == 0, 'reset'
            else:
                prev = prior[joint]
                assert x['q_pre_rad'] == prev['q_actual_rad'] and x['v_pre_rad_s'] == prev['v_actual_rad_s'] and x['previous_error_rad'] == prev['error_rad'], 'causal alignment'
                assert abs(x['ctrl_rad'] - prev['ctrl_rad']) <= .005 + 1e-12, 'command slew'
                assert abs(x['correction_rad'] - prev['correction_rad']) <= m.get('correction_rate_rad_per_tick', [0, 0, 0, 0])[joint] + 1e-12, 'correction rate'
                if tick <= m['command_ticks']:
                    squared += x['error_rad'] ** 2
                    n_command += 1
            peak = max(peak, abs(x['error_rad']))
            prior[joint] = x
            count += 1
    assert count == m['rows'] == (m['completed_ticks'] + 1) * m['dof'], 'row count/termination'
    assert abs(math.sqrt(squared / n_command) - m['metrics']['joint_rmse_rad']) < 1e-10 if n_command else m['metrics']['joint_rmse_rad'] == 0, 'RMSE recomputation'
    assert abs(peak - m['metrics']['peak_error_rad']) < 1e-10, 'peak recomputation'
    return m


def trajectory_key(m):

    values = m['initial_q_rad'] + m['goal_q_rad'] + [m['command_ticks'] * m['timestep_s']]
    return hashlib.sha256(json.dumps([m['dof']] + [round(x, 2) for x in values]).encode()).hexdigest()


def validate_dataset(path, full=True):
    identity = json.loads((path / 'campaign.json').read_text())
    groups, keys, seeds, partitions, categories, models = {}, {}, set(), {}, {}, []
    for folder in sorted(path.glob('e[0-9]*')):
        if not folder.is_dir():
            continue
        m = validate_episode(folder) if full else json.loads((folder / 'metadata.json').read_text())
        s = spec(m['index'], identity['config'])
        for field in ['family', 'seed', 'dof', 'category', 'severity', 'partition', 'episode']:
            assert m[field] == s[field], f'campaign spec {field}'
        assert m['source_tree_sha256'] == identity['source_tree_sha256'] and m['executable_sha256'] == identity['executable_sha256'], 'source identity'
        assert m['seed'] not in seeds, 'duplicate seed'
        seeds.add(m['seed'])
        for mapping, key in [(groups, m['family']), (keys, trajectory_key(m))]:
            assert mapping.get(key, m['partition']) == m['partition'], 'split leakage'
            mapping[key] = m['partition']
        partitions[m['partition']] = partitions.get(m['partition'], 0) + 1
        categories[m['category']] = categories.get(m['category'], 0) + 1
        models.append(m)
    for category in set(x['category'] for x in models) - {'nominal', 'recovery'}:
        changed = [m['physics'] for m in models if m['category'] == category]
        assert all(any(p[k] != default for k, default in [('gain_scale', 1), ('force_scale', 1), ('damping_scale', 1), ('friction_Nm', 0), ('mass_scale', 1), ('payload_kg', 0)]) for p in changed), 'no physical degradation'
    manifest = dict(schema_version=SCHEMA, episodes=len(models), families=len(groups), partitions=partitions,
                    categories=categories, episodes_manifest=[dict(episode=m['episode'], family=m['family'], seed=m['seed'], partition=m['partition'], trajectory_sha256=trajectory_key(m), trace_sha256=m['trace_sha256']) for m in models])
    atomic_json(path / 'split_manifest.json', manifest)
    return manifest


def source_hash():
    h = hashlib.sha256()
    for directory in ['src', 'include', 'adaptive', 'examples']:
        for base, dirs, files in os.walk(ROOT / directory):
            dirs[:] = sorted(d for d in dirs if d not in ['data', 'runs', '__pycache__'])
            for name in sorted(files):
                path = Path(base)/name
                if path.suffix in ['.cpp', '.hpp', '.py', '.cfg']:
                    h.update(str(path.relative_to(ROOT)).encode())
                    h.update(path.read_bytes())
    h.update((ROOT / 'CMakeLists.txt').read_bytes())
    return h.hexdigest()


def generate_one(exe, examples, destination, s, identity):
    target = destination / s['episode']
    if target.exists():
        m = validate_episode(target)
        assert m['executable_sha256'] == identity['executable_sha256'] and m['source_tree_sha256'] == identity['source_tree_sha256'], 'resume identity'
        return False, sum(p.stat().st_size for p in target.iterdir())

    with tempfile.TemporaryDirectory(prefix='.partial-', dir=destination) as temp:
        folder = Path(temp)
        args = [str(exe), str(examples), str(folder)] + [str(s[k]) for k in ['seed', 'family', 'dof', 'category', 'severity', 'partition', 'episode']]
        subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
        m = json.loads((folder / 'native.json').read_text())
        m.update(index=s['index'], executable_sha256=identity['executable_sha256'], source_tree_sha256=identity['source_tree_sha256'], source_dirty=identity['source_dirty'])
        with open(folder / 'telemetry.csv', 'rb') as inp, open(folder / 'telemetry.csv.gz', 'wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as gz:
                shutil.copyfileobj(inp, gz)
        (folder / 'telemetry.csv').unlink()
        m.update(trace_sha256=digest(folder / 'telemetry.csv.gz'), model_sha256=digest(folder / 'nominal.xml'))
        m['randomized_model_sha256'] = hashlib.sha256((m['model_sha256'] + json.dumps(m['physics'], sort_keys=True, separators=(',', ':'))).encode()).hexdigest()
        atomic_json(folder / 'metadata.json', m)
        (folder / 'native.json').unlink()
        validate_episode(folder)
        size = sum(p.stat().st_size for p in folder.iterdir())
        os.rename(folder, target)
    return True, size


def campaign(args):
    config = json.loads(args.config.read_text())
    workers = config['workers'] if args.workers is None else args.workers
    assert config['schema_version'] == SCHEMA and 1 <= workers <= 12
    assert config['categories'] == CATEGORIES and config['embodiments'] == [3, 4]
    count = config['episodes'] if args.episodes is None else args.episodes
    storage_budget = config['max_storage_gb'] if args.max_storage_gb is None else args.max_storage_gb
    assert math.isfinite(storage_budget) and storage_budget > 0, 'positive storage budget required'
    assert 1 <= count <= 1000000
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(config=config, executable_sha256=digest(args.exe), source_tree_sha256=source_hash(),
                        source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)))
        if (out / 'campaign.json').exists():
            saved = json.loads((out / 'campaign.json').read_text())
            assert saved['config'] == config and saved['executable_sha256'] == identity['executable_sha256'], 'campaign configuration or executable changed; use a new output directory'


            identity = saved
        else:
            atomic_json(out / 'campaign.json', identity)
        pinned = out / 'generator-pinned'
        if not pinned.exists():
            staged_binary = out / '.generator.tmp'
            shutil.copyfile(args.exe, staged_binary)
            assert digest(staged_binary) == identity['executable_sha256'], 'executable changed while pinning'
            staged_binary.chmod(0o755)
            os.replace(staged_binary, pinned)
        assert digest(pinned) == identity['executable_sha256'], 'pinned executable corrupted'
        began = time.perf_counter()
        generated, size = 0, sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        specs = (spec(i, config) for i in range(count))

        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            pending = set()
            exhausted = False
            while pending or not exhausted:
                while len(pending) < workers and not exhausted:
                    if size >= storage_budget * 1e9 or shutil.disk_usage(out).free < 100_000_000:
                        exhausted = True
                        break
                    s = next(specs, None)
                    if s is None:
                        exhausted = True
                        break
                    pending.add(pool.submit(generate_one, pinned.resolve(), args.examples.resolve(), out, s, identity))
                if not pending:
                    break
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                for result in done:
                    fresh, nbytes = result.result()
                    generated += fresh
                    if fresh:
                        size += nbytes
                    if generated and generated % 100 == 0:
                        print(f'{generated} generated, {generated/(time.perf_counter()-began):.2f} episodes/s', flush=True)
        elapsed = time.perf_counter() - began
        manifest = validate_dataset(out, full=False)
        report = dict(workers=workers, requested=count, completed=manifest['episodes'], newly_generated=generated, wall_s=elapsed,
                      new_episodes_per_s=generated / elapsed, storage_bytes=size, max_storage_gb=storage_budget, full_validation_per_episode=True,
                      partitions=manifest['partitions'], categories=manifest['categories'])
        atomic_json(out / 'generation_report.json', report)
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--exe', type=Path, required=True)
    gen.add_argument('--examples', type=Path, default=ROOT / 'examples')
    gen.add_argument('--config', type=Path, default=ROOT / 'adaptive/campaign.json')
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--episodes', type=int)
    gen.add_argument('--max-storage-gb', type=float, help='Explicit storage budget override; execution identity stays pinned')
    gen.add_argument('--workers', type=int, help='Defaults to campaign configuration')
    val = sub.add_parser('validate')
    val.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.command == 'generate':
        campaign(args)
    else:
        report = validate_dataset(args.path)
        print(json.dumps({k: v for k, v in report.items() if k != 'episodes_manifest'}, indent=2))

if __name__ == '__main__':
    main()
