from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

from .assets import ARM, FINGERS, MANIFEST
from .benchmark import BIN, COLORS, CONTROL_DT, SCENE_REVISION, finite, inside_bin
from .dataset import DOF, file_digest, read_json
from .panda import ARM as PANDA_ARM, FINGERS as PANDA_FINGERS
from .preflight import LAB_COMMIT, SIM
from .recording import validate_sample, validate_telemetry


@dataclass
class Episode:
    path: Path
    metadata: dict
    outcome: dict
    rows: list
    telemetry: list
    provenance: dict
    source_revision: str | None = None

    @property
    def duration(self):
        return self.rows[-1]['timestamp']

    def index(self, elapsed):
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid replay time')
        return max(0, bisect_right(self.rows, elapsed+1e-8, key=lambda row: row['timestamp'])-1)

    def state(self, elapsed):
        index = self.index(elapsed)
        row, physical = self.rows[index], self.telemetry[index]
        requested = [value+delta for value, delta in zip(row['ee_position'], row['action'][:3], strict=True)]
        ended = index == len(self.rows)-1
        tracking = None if ended else math.dist(self.rows[index+1]['ee_position'], requested)
        interventions = [{'timestamp': item['timestamp'], 'reason': reason}
                         for item in self.telemetry[:index+1] for reason in item['safety_events']]
        checkpoint = self.metadata.get('policy_service', {}).get('checkpoint_sha256')
        return {'embodiment': self.metadata['embodiment'], 'arm_dof': DOF[self.metadata['embodiment']],
                'controller': self.metadata['controller'], 'instruction': row['instruction'],
                'recorded_time': row['timestamp'], 'duration': self.duration, 'frame_index': index,
                'phase': row['phase'], 'ee_position': row['ee_position'], 'control_target': requested,
                'target_error_m': math.dist(row['ee_position'], requested),
                'requested_step_m': math.dist(row['ee_position'], requested),
                'tracking_error_m': tracking, 'applied_action': row['action'],
                'tracking_measurement': 'next recorded end effector versus current measured position plus command',
                'checkpoint_sha256': checkpoint, 'condition': self.metadata.get('condition', 'E0' if self.metadata['controller'] == 'classical_only' else 'unrecorded'),
                'dataset_sha256': self.metadata.get('dataset_sha256'),
                'source_revision': self.metadata.get('relic_commit') or self.source_revision,
                'intervention_timeline': interventions,
                'learned_evidence': self.metadata['controller'] != 'classical_only',
                'joint_position': row['joint_position'], 'joint_velocity': row['joint_velocity'],
                'cube_poses': physical['cube_poses'], 'contacts': physical['contacts'],
                'contact_forces': physical['contact_forces'], 'safety_events': physical['safety_events'],
                'unsafe': physical['unsafe'], 'fallback_triggers': self.outcome.get('fallback_triggers', []),
                'status': ('success' if self.outcome['success'] else 'failure') if ended else 'running',
                'failure_reason': self.outcome.get('failure_reason') if ended else None,
                'placement_error_xy_m': math.dist(physical['cube_poses'][self.metadata['target']][:2], BIN[:2]),
                'holding_final_state': elapsed > self.duration+1e-8}


def load_episode(path, expected_hashes=None, runtime_pin=None, source_sha256=None):
    path = Path(path).resolve()
    names = ('metadata.json', 'outcome.json', 'samples.jsonl', 'telemetry.jsonl')
    hashes = {name: file_digest(path/name) for name in names}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError('recorded trial differs from qualified evidence hashes')
    metadata, outcome = read_json(path/'metadata.json'), read_json(path/'outcome.json')
    robot = metadata.get('embodiment')
    if robot not in DOF or metadata.get('evidence_source') != 'isaac_lab_physx' or metadata.get('simulation_executed') is not True:
        raise ValueError('replay requires recorded physical Isaac Lab execution')
    if (metadata.get('runtime_pin') or runtime_pin) != {'isaacsim': SIM, 'isaaclab_commit': LAB_COMMIT, 'robot_commit': MANIFEST['commit']}:
        raise ValueError('recorded runtime differs from qualified pins')
    if metadata.get('physics_engine') != 'physx' or not (metadata.get('source_sha256') or source_sha256):
        raise ValueError('recorded physics or source provenance missing')
    if metadata.get('action_frame') != 'world' or metadata.get('action_alignment') != 'observation_to_next_control_tick':
        raise ValueError('recorded Cartesian action semantics differ')
    if metadata.get('scene_revision') != SCENE_REVISION:
        raise ValueError('recorded scene revision differs from renderer')
    if metadata.get('controller') not in ('classical_only', 'vla_only', 'fallback_assisted'):
        raise ValueError('unrecognized recorded controller')
    if metadata['controller'] != 'classical_only' and not metadata.get('policy_service', {}).get('checkpoint_sha256'):
        raise ValueError('learned replay lacks checkpoint provenance')
    expected_names = ARM+FINGERS if robot == 'openmanipulator_x' else PANDA_ARM+PANDA_FINGERS
    if metadata.get('joint_names') != list(expected_names):
        raise ValueError('recorded embodiment joint mapping differs')
    for key in ('seed', 'target'):
        if outcome.get(key) != metadata.get(key):
            raise ValueError('recorded outcome scenario mismatch')
    if outcome.get('controller') != metadata['controller'] or type(outcome.get('success')) is not bool:
        raise ValueError('recorded outcome controller or success mismatch')
    if metadata['target'] not in COLORS or outcome.get('simulation_executed') is not True:
        raise ValueError('invalid recorded task provenance')
    rows = [json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
    telemetry = [json.loads(line) for line in (path/'telemetry.jsonl').read_text().splitlines()]
    if not rows or len(rows) != len(telemetry) or (len(rows) == 1 and (outcome['success'] or not telemetry[0]['unsafe'] or outcome.get('failure_reason') != 'unsafe_movement')):
        raise ValueError('missing aligned replay telemetry')
    previous, lifted, wrong = -1., False, False
    for index, (row, physical) in enumerate(zip(rows, telemetry, strict=True)):
        previous = validate_sample(row, previous)
        validate_telemetry(physical)
        terminal_safety_stop = (index == len(rows)-1 and not outcome['success'] and physical['unsafe']
                                and outcome.get('failure_reason') == 'unsafe_movement'
                                and (index-1)*CONTROL_DT < previous <= index*CONTROL_DT+1e-5)
        if (abs(previous-index*CONTROL_DT) > 1e-5 and not terminal_safety_stop) or abs(previous-physical['timestamp']) > 1e-8:
            raise ValueError('replay timing or observation alignment mismatch')
        finite(row['joint_position'], DOF[robot]+2)
        finite(row['joint_velocity'], DOF[robot]+2)
        if row['instruction'] != f"Pick up the {metadata['target']} cube and place it in the bin.":
            raise ValueError('replay instruction differs from target')
        target = metadata['target']
        lifted |= physical['contacts'][target] and min(physical['contact_forces'][target]) > .005 and physical['cube_poses'][target][2] >= .055
        wrong |= any(pose[2] > .045 for color, pose in physical['cube_poses'].items() if color != target)
    if abs(outcome.get('duration', -1)-previous) > 1e-5 or rows[-1]['success'] != outcome['success'] or any(row['success'] for row in rows[:-1]):
        raise ValueError('replay terminal outcome or duration mismatch')
    if outcome['success'] and (not lifted or wrong or not inside_bin(telemetry[-1]['cube_poses'][metadata['target']])
            or outcome.get('pick_result') is not True or outcome.get('placement_result') is not True
            or any(item['unsafe'] or item['safety_events'] for item in telemetry)
            or telemetry[-1]['cube_poses'][metadata['target']] != outcome.get('final_object_pose')):
        raise ValueError('recorded success lacks physical grasp and placement evidence')
    from .completion import validate_completion
    validate_completion(metadata, outcome, rows, telemetry)
    return Episode(path, metadata, outcome, rows, telemetry, hashes)


def load_qualified_pair(seed, roots=None):
    if type(seed) is not int or not 0 <= seed < 20:
        raise ValueError('qualification replay seed must be between zero and nineteen')
    result = {}
    for robot in DOF:
        evidence = read_json(Path(__file__).with_name('qualification')/(robot+'.json'))
        root = Path(roots[robot]) if roots and robot in roots else Path(evidence['raw_artifact_root'])
        if file_digest(root/'report.json') != evidence['report_sha256']:
            raise ValueError('qualification report hash mismatch')
        reference = next(item for item in evidence['trials'] if item['seed'] == seed)
        result[robot] = load_episode(root/f'trial_{seed:02d}', reference['sha256'], evidence['runtime_pin'], evidence['source_sha256'])
        result[robot].source_revision = evidence.get('relic_commit')
    validate_pair(result)
    return result


def load_experiment_pair(root, index=0):
    root = Path(root).resolve()
    configuration = read_json(root/'experiment.json')
    if type(index) is not int or not 0 <= index < len(configuration['cases']):
        raise ValueError('unknown recorded experiment case')
    case = configuration['cases'][index]
    result = {}
    for robot in DOF:
        report = read_json(root/robot/'run/report.json')
        trial = report['trials'][index]
        folder = root/robot/'run'/f'trial_{index:04d}'
        expected = {key: value for key, value in trial['evidence_sha256'].items()
                    if key in ('metadata.json', 'outcome.json', 'samples.jsonl', 'telemetry.jsonl')}
        episode = load_episode(folder, expected)
        if episode.metadata['embodiment'] != robot or any(episode.metadata.get(key) != value for key, value in case.items()):
            raise ValueError('experiment replay differs from assigned case')
        if episode.metadata.get('source_sha256') != configuration['source_sha256']:
            raise ValueError('experiment replay source provenance differs')
        result[robot] = episode
    validate_pair(result)
    return result


def validate_pair(episodes):
    if set(episodes) != set(DOF):
        raise ValueError('both qualified embodiments required')
    left, right = episodes.values()
    for key in ('seed', 'target', 'purpose', 'split', 'scene_revision', 'perturbation', 'scenario_id'):
        if left.metadata.get(key) != right.metadata.get(key):
            raise ValueError('unmatched dual-view scenario: '+key)
    if left.rows[0]['instruction'] != right.rows[0]['instruction']:
        raise ValueError('unmatched task instructions')
    for color in COLORS:
        if math.dist(left.telemetry[0]['cube_poses'][color][:2], right.telemetry[0]['cube_poses'][color][:2]) > .001:
            raise ValueError('initial task conditions differ across embodiments')
    return max(episode.duration for episode in episodes.values())


class Playback:
    def __init__(self, duration, clock=time.monotonic):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('invalid playback duration')
        self.duration, self.clock = duration, clock
        self.elapsed, self.anchor, self.running = 0., clock(), False
        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            elapsed = min(self.duration, self.elapsed + (self.clock()-self.anchor if self.running else 0.))
            if elapsed >= self.duration:
                self.elapsed, self.running = self.duration, False
            return {'elapsed': elapsed, 'duration': self.duration, 'playing': self.running,
                    'mode': 'prerecorded', 'speed': 1.0}

    def command(self, action, position=None):
        with self.lock:
            elapsed = self.snapshot()['elapsed']
            if action in ('start', 'resume'):
                self.elapsed = 0. if elapsed >= self.duration else elapsed
                self.running = True
            elif action == 'pause':
                self.elapsed, self.running = elapsed, False
            elif action == 'reset':
                self.elapsed, self.running = 0., False
            elif action == 'seek':
                if type(position) not in (int, float) or not math.isfinite(position) or not 0 <= position <= self.duration:
                    raise ValueError('invalid seek position')
                self.elapsed, self.running = float(position), False
            else:
                raise ValueError('unknown playback command')
            self.anchor = self.clock()
            return self.snapshot()
