"""Qualify both robots on one clean, frozen revision without reusing legacy reports."""
import argparse
import json
from pathlib import Path
import sys
from .assets import MANIFEST
from .collection import atomic_json, execute, hashes, locked
from .dataset import DOF, read_json
from .preflight import LAB_COMMIT, SIM
from .qualify import campaign_cases, source_provenance
from .replay import load_episode, validate_pair


def qualify(root, python=sys.executable, run=False, timeout=2400):
    root = Path(root).resolve()
    provenance = source_provenance()
    configuration = {'schema': 'viper.common_qualification.v1', **provenance,
                     'cases': campaign_cases('qualification', 'train', 20),
                     'runtime_pin': {'isaacsim': SIM, 'isaaclab_commit': LAB_COMMIT, 'robot_commit': MANIFEST['commit']}}
    with locked(root):
        path = root/'qualification.json'
        if path.exists() and read_json(path) != configuration:
            raise ValueError('qualification source revision or frozen inputs changed')
        atomic_json(path, configuration)
        result = {'schema': 'viper.common_qualification_result.v1', 'status': 'NOT_EXECUTED',
                  'common_revision': provenance['relic_commit'], 'simulation_executed': False,
                  'robots': {}, 'errors': [], 'commands': {}}
        for robot in DOF:
            command = [str(python), '-m', 'embodiment.qualify', '--mode', 'qualification', '--robot', robot,
                       '--output', str(root/robot/'run'), '--headless', '--device', 'cuda:0']
            result['commands'][robot] = command
        if not run:
            atomic_json(root/'result.json', result)
            return result
        if provenance['relic_worktree_dirty']:
            raise ValueError('qualification requires a clean committed revision')
        try:
            episodes = {}
            for robot, command in result['commands'].items():
                job = root/robot
                job.mkdir(exist_ok=True)
                if not (job/'run/report.json').exists() and not (job/'process.log').exists():
                    execute(command, job, timeout)
                report = read_json(job/'run/report.json')
                if (report.get('relic_commit') != provenance['relic_commit'] or report.get('source_sha256') != provenance['source_sha256']
                        or report.get('runtime_pin') != configuration['runtime_pin'] or report.get('source_changed_during_run') != []
                        or report.get('status') != 'PASS' or report.get('completed') is not True
                        or report.get('simulation_executed') is not True or report.get('embodiment') != robot
                        or [dict(seed=row['seed'], target=row['target']) for row in report['trials']] != configuration['cases']):
                    raise ValueError('qualification report fails frozen revision, scenario or physical gate: '+robot)
                episodes[robot] = [load_episode(job/'run'/f'trial_{seed:02d}') for seed in range(20)]
                for episode in episodes[robot]:
                    if episode.metadata.get('source_sha256') != provenance['source_sha256']:
                        raise ValueError('trial source differs from frozen qualification')
                result['robots'][robot] = {'report': report, 'evidence_sha256': hashes(job/'run')}
            for index in range(20): validate_pair({robot: episodes[robot][index] for robot in DOF})
            if source_provenance() != provenance:
                raise ValueError('source changed during common qualification')
            result.update(status='PASS', simulation_executed=True)
        except Exception as error:
            result.update(status='BLOCKED')
            result['errors'].append(f'{type(error).__name__}: {error}')
        atomic_json(root/'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = qualify(args.root, args.python, args.execute)
    print(json.dumps(result, indent=2))
    return 2 if result['status'] == 'BLOCKED' else 0


if __name__ == '__main__': raise SystemExit(main())
