import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import os
import shutil
import tempfile
from .benchmark import COLORS, CONTROL_DT, SCENE_REVISION, TASK_GRIPPER_LIMITS, layout
from .recording import validate_sample, validate_telemetry
from .scenarios import HELD_OUT, identity, region, validate_scenario

GROOT_COMMIT = '944fe70c4fd54e3fcb80a7d7258a328f9c061176'
VIDEO_KEY = 'observation.images.overhead'
DOF = {'openmanipulator_x': 4, 'franka_panda': 7}


def read_json(path: Path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


@dataclass
class Episode:
    path: Path
    metadata: dict
    outcome: dict
    rows: list[dict]
    shape: tuple[int, int, int]
    fingerprint: str


def load_episode(path: Path, require_physics: bool = True, expert_only: bool = True) -> Episode:
    from PIL import Image
    path = Path(path).resolve()
    metadata, outcome = read_json(path/'metadata.json'), read_json(path/'outcome.json')
    robot = metadata.get('embodiment')
    if robot not in DOF:
        raise ValueError('unknown embodiment')
    allowed_controllers = ('classical_only',) if expert_only else ('classical_only', 'vla_only', 'fallback_assisted')
    if require_physics and (metadata.get('simulation_executed') is not True or metadata.get('physics_engine') != 'physx' or metadata.get('controller') not in allowed_controllers or metadata.get('evidence_source') != 'isaac_lab_physx'):
        raise ValueError('physical scripted demonstration evidence required')
    if require_physics and metadata.get('controller') != 'classical_only' and not metadata.get('policy_service', {}).get('checkpoint_sha256'):
        raise ValueError('learned evaluation requires checkpoint provenance')
    if type(outcome.get('success')) is not bool:
        raise ValueError('missing episode outcome')
    target, seed = metadata.get('target'), metadata.get('seed')
    if target not in COLORS or type(seed) is not int or seed < 0:
        raise ValueError('missing scenario identity')
    if require_physics:
        from .preflight import LAB_COMMIT, SIM
        from .assets import MANIFEST
        expected_runtime = {'isaacsim': SIM, 'isaaclab_commit': LAB_COMMIT, 'robot_commit': MANIFEST['commit']}
        if metadata.get('runtime_pin') != expected_runtime or not metadata.get('source_sha256') or not metadata.get('relic_commit'):
            raise ValueError('pinned simulator and source provenance required')
        if metadata.get('scene_revision') != SCENE_REVISION:
            raise ValueError('scene/control calibration revision mismatch')
        from .assets import ARM, FINGERS
        from .panda import ARM as PANDA_ARM, FINGERS as PANDA_FINGERS
        expected_names = ARM+FINGERS if robot == 'openmanipulator_x' else PANDA_ARM+PANDA_FINGERS
        if metadata.get('joint_names') != list(expected_names):
            raise ValueError('embodiment joint names do not match the qualified adapter')
        if outcome.get('robot') != robot or outcome.get('seed') != seed or outcome.get('target') != target:
            raise ValueError('outcome identity mismatch')
        if outcome.get('simulation_executed') is not True or outcome.get('controller') != metadata.get('controller'):
            raise ValueError('outcome provenance mismatch')
        if outcome['success'] and (outcome.get('pick_result') is not True or outcome.get('placement_result') is not True):
            raise ValueError('successful demonstration lacks physical pick-and-placement evidence')
    joint_count = DOF[robot] + 2
    if len(metadata.get('joint_names', [])) != joint_count or len(set(metadata['joint_names'])) != joint_count:
        raise ValueError('joint mapping mismatch')
    if metadata.get('action_frame') != 'world' or metadata.get('action_fields') != ['delta_x','delta_y','delta_z','gripper']:
        raise ValueError('action semantics mismatch')
    if (metadata.get('action_alignment') != 'observation_to_next_control_tick'
            or metadata.get('gripper_range') != [0, 1] or metadata.get('gripper_open') != 1
            or metadata.get('time_unit') != 'second' or metadata.get('position_unit') != 'meter'):
        raise ValueError('action alignment or unit convention mismatch')
    if not math.isclose(metadata.get('control_dt', 0), CONTROL_DT, abs_tol=1e-9):
        raise ValueError('control frequency mismatch')
    rows, shape, previous = [], None, -1.
    for line in (path/'samples.jsonl').read_text().splitlines():
        row = json.loads(line)
        timestamp = validate_sample(row, previous)
        if not math.isclose(timestamp, len(rows)*CONTROL_DT, abs_tol=1e-5):
            raise ValueError('nonuniform or misaligned timestamps')
        previous = timestamp
        if len(row['joint_position']) != joint_count:
            raise ValueError('state dimensionality mismatch')
        if row['instruction'] != f'Pick up the {target} cube and place it in the bin.':
            raise ValueError('language annotation mismatch')
        if row['frame'] is None:
            raise ValueError('camera frame missing')
        if row['frame'] != f'camera/{len(rows):06d}.png':
            raise ValueError('noncanonical or reused camera frame')
        frame = (path/row['frame']).resolve()
        if not frame.is_relative_to(path):
            raise ValueError('camera path escapes episode')
        with Image.open(frame) as image:
            if image.format != 'PNG' or image.mode != 'RGB':
                raise ValueError('expected RGB PNG camera frames')
            current = (image.height, image.width, 3)
            image.verify()
        if shape is not None and shape != current:
            raise ValueError('camera shape changed within episode')
        shape = current
        rows.append(row)
    if len(rows) < 2 or any(row['success'] for row in rows[:-1]) or rows[-1]['success'] != outcome['success']:
        raise ValueError('invalid episode boundary or outcome alignment')
    if require_physics:
        from .benchmark import inside_bin
        telemetry = [json.loads(line) for line in (path/'telemetry.jsonl').read_text().splitlines()]
        if len(telemetry) != len(rows) or any(abs(a['timestamp']-b['timestamp']) > 1e-8 for a,b in zip(rows,telemetry,strict=True)):
            raise ValueError('physical telemetry is not aligned with demonstrations')
        for item in telemetry:
            validate_telemetry(item)
        lifted_with_contact = any(item['contacts'].get(target) is True
                                  and len(item.get('contact_forces', {}).get(target, [])) == 2
                                  and all(type(force) in (int, float) and math.isfinite(force) and force > .005
                                          for force in item['contact_forces'][target])
                                  and item['cube_poses'][target][2] >= .055 for item in telemetry)
        if outcome['success']:
            if not lifted_with_contact or any(item['unsafe'] or item['safety_events'] for item in telemetry):
                raise ValueError('physical contact/lift or safety evidence does not support success')
            if any(pose[2] > .045 for item in telemetry for color,pose in item['cube_poses'].items() if color != target):
                raise ValueError('wrong-color lift cannot be labeled successful')
            final_pose = telemetry[-1]['cube_poses'][target]
            if not inside_bin(final_pose) or final_pose != outcome.get('final_object_pose'):
                raise ValueError('placement evidence does not support success')
        from .completion import validate_completion
        validate_completion(metadata, outcome, rows, telemetry)
    fingerprint = identity(seed, target)
    if metadata.get('scenario_id', fingerprint) != fingerprint:
        raise ValueError('scenario fingerprint mismatch')
    return Episode(path, metadata, outcome, rows, shape, fingerprint)


def validate_collection(paths: list[Path], require_physics: bool = True) -> tuple[list[Episode], dict]:
    if not paths:
        raise ValueError('empty collection')
    episodes = [load_episode(path, require_physics) for path in paths]
    seen = set()
    scenario_splits = {}
    for episode in episodes:
        meta = episode.metadata
        key = (meta['embodiment'], episode.fingerprint)
        if key in seen:
            raise ValueError('duplicate episode')
        seen.add(key)
        split = meta.get('split')
        if split not in ('train', 'validation', 'evaluation') or meta.get('purpose') != 'collection':
            raise ValueError('qualification or unassigned episodes cannot enter a dataset')
        if meta['seed'] in range(20):
            raise ValueError('qualification seed leakage')
        validate_scenario(meta)
        pair = (meta['target'], region(layout(meta['seed'])[meta['target']]))
        if (pair in HELD_OUT) != (split == 'evaluation'):
            raise ValueError('held-out configuration leakage')
        old = scenario_splits.setdefault(episode.fingerprint, split)
        if old != split:
            raise ValueError('train/test scenario leakage')
    report = {'schema': 'relic.collection_validation.v1', 'episodes': len(episodes),
              'frames': sum(len(ep.rows) for ep in episodes),
              'physical_evidence_required': require_physics,
              'successful_episodes': sum(ep.outcome['success'] for ep in episodes),
              'embodiments': dict(Counter(ep.metadata['embodiment'] for ep in episodes)),
              'splits': dict(Counter(ep.metadata['split'] for ep in episodes)),
              'colors': dict(Counter(ep.metadata['target'] for ep in episodes))}
    return episodes, report


def modality(joints: int) -> dict:
    return {'state': {'joint_position': {'start': 0, 'end': joints},
                      'joint_velocity': {'start': joints, 'end': 2*joints},
                      'ee_position': {'start': 2*joints, 'end': 2*joints+3},
                      'gripper': {'start': 2*joints+3, 'end': 2*joints+4}},
            'action': {'delta_position': {'start': 0, 'end': 3}, 'gripper': {'start': 3, 'end': 4}},
            'video': {'overhead': {'original_key': VIDEO_KEY}},
            'annotation': {'human.task_description': {'original_key': 'annotation.human.task_description'}}}


def state(row: dict, robot: str) -> list[float]:
    closed,opened = TASK_GRIPPER_LIMITS[robot]
    aperture = max(0., min(1., (row['joint_position'][-2] - closed)/(opened - closed)))
    return row['joint_position'] + row['joint_velocity'] + row['ee_position'] + [aperture]


def export(paths: list[Path], output: Path, require_physics: bool = True) -> dict:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        info = _export(paths, staging/'dataset', require_physics)
        validate_export(staging/'dataset', require_physics)
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging/'dataset', output)
        return info
    finally:
        shutil.rmtree(staging)


def _export(paths: list[Path], output: Path, require_physics: bool = True) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    episodes, report = validate_collection(paths, require_physics)
    robots = {ep.metadata['embodiment'] for ep in episodes}
    splits = {ep.metadata['split'] for ep in episodes}
    shapes = {ep.shape for ep in episodes}
    if len(robots) != 1 or len(splits) != 1 or len(shapes) != 1:
        raise ValueError('export each embodiment and split separately with one camera shape')
    if any(not ep.outcome['success'] for ep in episodes):
        raise ValueError('failed attempts remain diagnostic records, not expert training demonstrations')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = output/'meta'
    meta.mkdir()
    robot, split, shape = next(iter(robots)), next(iter(splits)), next(iter(shapes))
    tasks = sorted({row['instruction'] for ep in episodes for row in ep.rows})
    task_ids = {task: i for i, task in enumerate(tasks)}
    episode_meta, source_meta, states, actions = [], [], [], []
    index = 0
    for episode_index, episode in enumerate(episodes):
        chunk = episode_index//1000
        parquet = output/f'data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet'
        video = output/f'videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{episode_index:06d}.mp4'
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        frames = [episode.path/row['frame'] for row in episode.rows]
        if any(frame != episode.path/f'camera/{i:06d}.png' for i, frame in enumerate(frames)):
            raise ValueError('noncanonical camera sequence')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','30','-i',str(episode.path/'camera/%06d.png'),
                        '-frames:v',str(len(frames)),'-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(video)],
                       check=True, capture_output=True, timeout=300)
        columns = []
        for frame_index, row in enumerate(episode.rows):
            values = state(row, robot)
            states.append(values)
            actions.append(row['action'])
            task_id = task_ids[row['instruction']]
            columns.append({'observation.state': values, 'action': row['action'], 'timestamp': row['timestamp'],
                            'frame_index': frame_index, 'episode_index': episode_index, 'index': index,
                            'task_index': task_id, 'annotation.human.task_description': task_id,
                            'next.done': frame_index == len(episode.rows)-1,
                            'next.reward': float(frame_index == len(episode.rows)-1), 'next.success': True})
            index += 1
        table = pa.Table.from_pylist(columns)
        for key in ('timestamp','next.reward'):
            table = table.set_column(table.schema.get_field_index(key),key,pa.array([r[key] for r in columns],type=pa.float32()))
        for key in ('observation.state','action'):
            table = table.set_column(table.schema.get_field_index(key), key, pa.array([r[key] for r in columns], type=pa.list_(pa.float32())))
        pq.write_table(table, parquet)
        episode_meta.append({'episode_index': episode_index, 'tasks': [episode.rows[0]['instruction']], 'length': len(columns)})
        source_meta.append({'episode_index': episode_index, 'scenario_id': episode.fingerprint, 'seed': episode.metadata['seed'],
                            'target': episode.metadata['target'], 'split': split,
                            'recording_sha256': hashlib.sha256((episode.path/'samples.jsonl').read_bytes()).hexdigest(),
                            'metadata': episode.metadata, 'outcome': episode.outcome})
        source_meta[-1]['export_sha256'] = {str(file.relative_to(output)): file_digest(file)
                                           for file in (parquet, video)}
    features = {key: {'dtype': 'float32', 'shape': [len(values[0])], 'names': None}
                for key, values in (('observation.state', states), ('action', actions))}
    for key in ('timestamp','next.reward'):
        features[key] = {'dtype': 'float32', 'shape': [1], 'names': None}
    for key in ('frame_index','episode_index','index','task_index','annotation.human.task_description'):
        features[key] = {'dtype': 'int64', 'shape': [1], 'names': None}
    for key in ('next.done','next.success'):
        features[key] = {'dtype': 'bool', 'shape': [1], 'names': None}
    features[VIDEO_KEY] = {'dtype': 'video', 'shape': list(shape), 'names': ['height','width','channels'],
                           'info': {'video.fps': 30, 'video.height': shape[0], 'video.width': shape[1],
                                    'video.channels': 3, 'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                                    'video.is_depth_map': False, 'has_audio': False}}
    info = {'codebase_version': 'v2.1', 'robot_type': robot, 'total_episodes': len(episodes), 'total_frames': index,
            'total_tasks': len(tasks), 'total_videos': len(episodes), 'total_chunks': (len(episodes)+999)//1000,
            'chunks_size': 1000, 'fps': 30, 'splits': {split: f'0:{len(episodes)}'}, 'features': features,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'relic': {'physical_evidence_validated': require_physics, 'groot_source_commit': GROOT_COMMIT,
                      'action_semantics': 'world_meters_per_control_tick_and_absolute_gripper', 'scene_revision': SCENE_REVISION}}
    write_json(meta/'info.json', info)
    write_json(meta/'modality.json', modality(DOF[robot]+2))
    for name, rows in (('episodes', episode_meta), ('tasks', [{'task_index': i,'task': task} for i,task in enumerate(tasks)]), ('relic_sources', source_meta)):
        (meta/f'{name}.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    stats = {}
    for key, values in (('observation.state',states),('action',actions)):
        array = np.asarray(values, dtype=np.float32)
        stats[key] = {name: getattr(array,name)(axis=0).tolist() for name in ('min','max','mean','std')}
        stats[key]['count'] = [len(array)]
        for q in (.01,.05,.95,.99):
            stats[key][f'q{int(q*100):02d}'] = np.quantile(array,q,axis=0).tolist()
    write_json(meta/'stats.json', stats)
    write_json(meta/'relic_validation.json', report)
    return info


def validate_export(path: Path, require_physics: bool = True) -> dict:
    import av
    import numpy as np
    import pyarrow.parquet as pq
    path = Path(path)
    info = read_json(path/'meta/info.json')
    if info.get('codebase_version') != 'v2.1' or info.get('robot_type') not in DOF or info.get('fps') != 30:
        raise ValueError('unsupported LeRobot runtime schema')
    if require_physics and info.get('relic',{}).get('physical_evidence_validated') is not True:
        raise ValueError('fixture or unverified data cannot pass physical dataset validation')
    robot = info['robot_type']
    count = DOF[robot]+2
    if (info.get('chunks_size') != 1000
            or info.get('data_path') != 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
            or info.get('video_path') != 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'):
        raise ValueError('noncanonical dataset paths or chunk size')
    if (info.get('relic', {}).get('groot_source_commit') != GROOT_COMMIT
            or info['relic'].get('scene_revision') != SCENE_REVISION
            or info['relic'].get('action_semantics') != 'world_meters_per_control_tick_and_absolute_gripper'):
        raise ValueError('export source or action semantics mismatch')
    features = info.get('features', {})
    if (features.get('observation.state', {}).get('shape') != [2*count+4]
            or features.get('action', {}).get('shape') != [4]):
        raise ValueError('feature dimensionality mismatch')
    if read_json(path/'meta/modality.json') != modality(count):
        raise ValueError('GR00T modality mapping mismatch')
    def lines(name):
        return [json.loads(line) for line in (path/f'meta/{name}.jsonl').read_text().splitlines()]
    episodes, tasks, sources = lines('episodes'), lines('tasks'), lines('relic_sources')
    if not episodes or len(episodes) != info['total_episodes'] or len(sources) != len(episodes):
        raise ValueError('episode metadata count mismatch')
    if info.get('total_videos') != len(episodes) or info.get('total_chunks') != (len(episodes)+999)//1000:
        raise ValueError('video or chunk metadata count mismatch')
    task_map = {row['task_index']: row['task'] for row in tasks}
    if len(task_map) != len(tasks) or len(tasks) != info['total_tasks']:
        raise ValueError('task metadata mismatch')
    split_names = list(info.get('splits',{}))
    if len(split_names) != 1 or split_names[0] not in ('train','validation','evaluation'):
        raise ValueError('invalid dataset split')
    split = split_names[0]
    if info['splits'][split] != f'0:{len(episodes)}':
        raise ValueError('dataset split indices mismatch')
    seen, all_states, all_actions = set(), [], []
    offset = 0
    for index, (episode,source) in enumerate(zip(episodes,sources,strict=True)):
        if episode['episode_index'] != index or source['episode_index'] != index:
            raise ValueError('episode indices do not align')
        metadata = source['metadata']
        validate_scenario(metadata)
        if (metadata.get('embodiment') != robot or metadata.get('purpose') != 'collection'
                or any(metadata.get(key) != source.get(key) for key in ('seed', 'target', 'split', 'scenario_id'))):
            raise ValueError('source metadata identity mismatch')
        instruction = f"Pick up the {source['target']} cube and place it in the bin."
        if episode['tasks'] != [instruction]:
            raise ValueError('episode instruction differs from target color')
        fingerprint = identity(source['seed'],source['target'])
        if source['scenario_id'] != fingerprint or fingerprint in seen or source['split'] != split:
            raise ValueError('duplicate scenario or split mismatch')
        seen.add(fingerprint)
        if source['seed'] in range(20):
            raise ValueError('qualification seed leakage')
        pair = source['target'],region(layout(source['seed'])[source['target']])
        if (pair in HELD_OUT) != (split == 'evaluation'):
            raise ValueError('held-out configuration leakage')
        if require_physics and (source['metadata'].get('simulation_executed') is not True or source['outcome'].get('success') is not True):
            raise ValueError('unverified physical source')
        parquet_path = path/info['data_path'].format(episode_chunk=index//1000,episode_index=index)
        video_path = path/info['video_path'].format(episode_chunk=index//1000,episode_index=index,video_key=VIDEO_KEY)
        expected_hashes = {str(file.relative_to(path)): file_digest(file) for file in (parquet_path, video_path)}
        if source.get('export_sha256') != expected_hashes:
            raise ValueError('exported data or video checksum mismatch')
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        length = episode['length']
        if len(rows) != length or length < 2:
            raise ValueError('episode frame count mismatch')
        state_values = np.asarray([row['observation.state'] for row in rows])
        action_values = np.asarray([row['action'] for row in rows])
        if state_values.shape != (length,2*count+4) or action_values.shape != (length,4):
            raise ValueError('state or action dimensionality mismatch')
        if not np.isfinite(state_values).all() or not np.isfinite(action_values).all():
            raise ValueError('nonfinite dataset values')
        for frame_index,row in enumerate(rows):
            if row['index'] != offset+frame_index or row['episode_index'] != index or row['frame_index'] != frame_index:
                raise ValueError('global and episode boundaries do not align')
            if not math.isclose(row['timestamp'],frame_index/30,abs_tol=1e-4):
                raise ValueError('frame timestamp mismatch')
            terminal = frame_index == length-1
            if row['next.done'] is not terminal or row['next.success'] is not True or row['next.reward'] != float(terminal):
                raise ValueError('terminal label mismatch')
            task_id = row['task_index']
            if task_id != row['annotation.human.task_description'] or task_id not in task_map or task_map[task_id] not in episode['tasks']:
                raise ValueError('language index mismatch')
            from .benchmark import Action
            Action(tuple(row['action'][:3]),row['action'][3])
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if float(stream.average_rate) != 30:
                raise ValueError('video rate mismatch')
            decoded = 0
            for frame in container.decode(stream):
                if [frame.height,frame.width,3] != info['features'][VIDEO_KEY]['shape']:
                    raise ValueError('video dimensions mismatch')
                if not math.isclose(float(frame.time),decoded/30,abs_tol=1e-5):
                    raise ValueError('video timestamp mismatch')
                decoded += 1
            if decoded != length:
                raise ValueError('video and state modality alignment mismatch')
        all_states.extend(state_values)
        all_actions.extend(action_values)
        offset += length
    if offset != info['total_frames']:
        raise ValueError('total frame count mismatch')
    statistics = read_json(path/'meta/stats.json')
    for key,values in (('observation.state',all_states),('action',all_actions)):
        array = np.asarray(values,dtype=np.float32)
        expected = {field: getattr(array, field)(axis=0) for field in ('min', 'max', 'mean', 'std')}
        expected.update({f'q{int(q*100):02d}': np.quantile(array, q, axis=0) for q in (.01, .05, .95, .99)})
        expected['count'] = np.asarray([len(array)])
        for field, computed in expected.items():
            actual = np.asarray(statistics[key][field])
            if actual.shape != computed.shape or not np.isfinite(actual).all() or not np.allclose(computed,actual,rtol=1e-5,atol=1e-7):
                raise ValueError('normalization statistics mismatch')
    return {'schema':'relic.lerobot_validation.v1','episodes':len(episodes),'frames':offset,
            'embodiment':robot,'split':split,'video_decode_verified':True,
            'physical_evidence_required':require_physics,'groot_source_commit':GROOT_COMMIT}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episodes', type=Path, nargs='*')
    parser.add_argument('--validate-export',type=Path)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    if args.validate_export:
        result = validate_export(args.validate_export)
    elif args.export:
        result = export(args.episodes, args.export)
    else:
        _, result = validate_collection(args.episodes)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
