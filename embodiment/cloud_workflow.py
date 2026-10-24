"""Generate or execute one bounded approved-host stage; no provisioning or downloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .collection import atomic_json, locked, execute
from .dataset import read_json
from .qualify import source_provenance

STAGES = ('interfaces', 'simulation_host', 'contact', 'qualification', 'manipulation', 'baseline',
          'pilot', 'training_host', 'g3_preflight', 'g3',
          'train_a', 'train_b_only', 'train_a_to_b', 'evaluate_a', 'evaluate_b_only', 'evaluate_a_to_b',
          'compare', 'artifacts')


def commands(root, simulation_python, training_python, isaac_source, groot_source, pretrained, reader_library_path=None):
    root, isaac_source, groot_source, pretrained = (Path(path).resolve() for path in (root, isaac_source, groot_source, pretrained))
    sim, train = str(simulation_python), str(training_python)
    pilot = root/'pilot'
    extra = ['--reader-library-path', str(reader_library_path)] if reader_library_path else []
    smoke = [train, '-m', 'embodiment.cross_smoke', '--source', str(groot_source),
             '--dataset-a', str(pilot/'exports/openmanipulator_x/train'), '--dataset-b', str(pilot/'exports/franka_panda/train'),
             '--pretrained', str(pretrained), '--output', str(root/'g3')]
    planned = {
        'interfaces': [train, '-m', 'embodiment.isaac_interfaces', '--source', str(isaac_source)],
        'simulation_host': [sim, '-m', 'embodiment.host', '--stage', 'simulation', '--volume', str(root)],
        'contact': [sim, '-m', 'embodiment.contact_probe', '--output', str(root/'contact'), '--headless', '--device', 'cuda:0'],
        'qualification': [sim, '-m', 'embodiment.qualification_pair', '--root', str(root/'qualification'), '--python', sim, '--execute'],
        'manipulation': [train, '-m', 'embodiment.manipulation', '--root', str(root/'manipulation'), '--python', sim,
                         '--starts', 'home', 'left', 'right', '--perturbed', '--execute'],
        'baseline': [train, '-m', 'embodiment.experiment', '--root', str(root/'baseline'), '--suite', 'heldout',
                     '--scenarios', '50', '--python', sim, '--timeout', '3600'],
        'pilot': [train, '-m', 'embodiment.pilot', '--root', str(pilot), '--episodes', '12', '--python', sim,
                  '--data-python', train, '--groot-source', str(groot_source), *extra, '--execute'],
        'training_host': [train, '-m', 'embodiment.cloud', '--source', str(groot_source), '--pretrained', str(pretrained), '--volume', str(root)],
        'g3_preflight': smoke,
        'g3': [*smoke, '--execute'],
        'compare': [train, '-m', 'embodiment.campaign_report', '--root', str(root/'campaign'), '--reference', str(root/'baseline')],
        'artifacts': [train, '-m', 'embodiment.artifacts', '--root', str(root), '--output', str(root.parent/(root.name+'.tar.gz'))],
    }
    for stage in STAGES:
        if stage.startswith(('train_', 'evaluate_')):
            planned[stage] = [train, '-m', 'embodiment.training_campaign', '--source', str(groot_source),
                '--dataset-a', str(pilot/'exports/openmanipulator_x/train'), '--dataset-b', str(pilot/'exports/franka_panda/train'),
                '--pretrained', str(pretrained), '--root', str(root/'campaign'), '--smoke-root', str(root/'g3'),
                '--simulation-python', sim, '--stage', stage, '--execute']
    return planned


def run(root, simulation_python, training_python, isaac_source, groot_source, pretrained,
        stage=None, execute_stage=False, reader_library_path=None, panda_usd=None, robot_cache=None):
    root = Path(root).resolve()
    planned = commands(root, simulation_python, training_python, isaac_source, groot_source, pretrained, reader_library_path)
    with locked(root/'workflow'):
        configuration = {'schema': 'viper.cloud_workflow.v1', 'commands': planned, 'source': source_provenance()}
        if panda_usd is not None: configuration['panda_usd'] = str(Path(panda_usd).resolve())
        if robot_cache is not None: configuration['robot_cache'] = str(Path(robot_cache).resolve())
        path = root/'workflow/configuration.json'
        if path.exists() and read_json(path) != configuration: raise ValueError('workflow source or commands changed')
        atomic_json(path, configuration)
        result = {'schema': 'viper.cloud_workflow_result.v1', 'status': 'NOT_EXECUTED',
                  'commands': planned, 'provisions_hardware': False, 'downloads_models': False}
        if not execute_stage: return result
        if stage not in STAGES: raise ValueError('select an explicit workflow stage')
        if configuration['source']['relic_worktree_dirty']: raise ValueError('approved-host stages require a clean committed revision')
        checks = []
        if stage in ('contact', 'qualification', 'pilot', 'manipulation', 'baseline') or stage.startswith('evaluate_'):
            checks.append(planned['simulation_host'])
        if stage in ('pilot', 'g3_preflight', 'g3', 'manipulation', 'baseline') or stage.startswith(('train_', 'evaluate_')):
            if read_json(root/'qualification/result.json')['status'] != 'PASS': raise ValueError('common-revision qualification required')
            if read_json(root/'qualification/qualification.json')['source_sha256'] != configuration['source']['source_sha256']:
                raise ValueError('qualification source differs from this workflow')
        if stage in ('g3_preflight', 'g3') or stage.startswith(('train_', 'evaluate_')):
            pilot = read_json(root/'pilot/execution/result.json')
            if pilot.get('training_ready') is not True or pilot.get('status') != 'VALIDATED': raise ValueError('verified physical pilot required before G3')
            if read_json(root/'pilot/pilot.json')['source_sha256'] != configuration['source']['source_sha256']:
                raise ValueError('pilot source differs from this workflow')
            checks.append(planned['training_host'])
        for command in checks:
            checked = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if checked.returncode: raise RuntimeError('host gate failed: '+checked.stdout[-2000:]+checked.stderr[-1000:])
        job = root/'workflow'/stage
        job.mkdir(exist_ok=True)
        attempt = job/f'attempt-{len(list(job.glob("attempt-*"))):04d}'
        attempt.mkdir()
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
        if robot_cache is not None: environment['VIPER_ASSET_CACHE'] = str(Path(robot_cache).resolve())
        if panda_usd is not None:
            from .dataset import file_digest
            usd = Path(panda_usd).resolve()
            from .asset_cache import PANDA
            asset_root = usd.parents[len(Path(PANDA).parts)-1]
            assets = read_json(asset_root/'assets.json')
            if str(usd) != assets['usd']: raise ValueError('Panda cache receipt differs from selected USD')
            for name, item in assets['files'].items():
                if file_digest(asset_root/name) != item['sha256']: raise ValueError('Panda USD cache integrity failure')
            environment['VIPER_PANDA_USD'] = str(usd)
        if reader_library_path and (stage in ('g3', 'g3_preflight') or stage.startswith(('train_', 'evaluate_'))):
            environment['LD_LIBRARY_PATH'] = str(reader_library_path)
        timeout = 15000 if stage.startswith(('train_', 'evaluate_')) else (7200 if stage in ('pilot', 'baseline', 'manipulation') else 3600)
        measured = execute(planned[stage], attempt, timeout, environment)
        result.update(status='PROCESS_COMPLETED' if measured['returncode'] == 0 else 'BLOCKED', stage=stage, process=measured,
                      claim='process status only; qualification, pilot and G3 retain their own evidence gates')
        atomic_json(job/'result.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'isaac-source', 'groot-source', 'pretrained'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--simulation-python', required=True)
    parser.add_argument('--training-python', required=True)
    parser.add_argument('--reader-library-path', type=Path)
    parser.add_argument('--panda-usd', type=Path)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run(args.root, args.simulation_python, args.training_python, args.isaac_source, args.groot_source,
                 args.pretrained, args.stage, args.execute, args.reader_library_path, args.panda_usd)
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result['status'] == 'BLOCKED' else 0)
