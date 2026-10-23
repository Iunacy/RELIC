import argparse
import json
from pathlib import Path
import subprocess
import sys
from .dataset import DOF, GROOT_COMMIT, read_json


def verify_source(source: Path) -> None:
    source = Path(source).resolve()
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit != GROOT_COMMIT:
        raise ValueError('GR00T checkout differs from the verified source revision')
    changes = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all','--','gr00t'],text=True)
    if changes:
        raise ValueError('GR00T runtime source has local modifications')


def validate_with_upstream(source: Path, dataset: Path, decode_video: bool = False) -> dict:
    verify_source(source)
    sys.path.insert(0,str(Path(source).resolve()))
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    import numpy as np
    from .groot_modalities import RELIC_CONFIG
    info = read_json(Path(dataset)/'meta/info.json')
    loader = LeRobotEpisodeLoader(dataset,RELIC_CONFIG)
    joints = DOF[info['robot_type']]+2
    frames, samples = 0, 0
    tag = EmbodimentTag.NEW_EMBODIMENT
    processor = StateActionProcessor({tag.value: RELIC_CONFIG},
        {tag.value: loader.get_dataset_statistics()}, use_percentiles=True, use_relative_action=True)
    processor.eval()
    for index,episode in enumerate(loader.episodes_metadata):
        data = loader[index] if decode_video else loader._load_parquet_data(index)
        if len(data) != episode['length']:
            raise ValueError('upstream loader frame count mismatch')
        expected = {'state.joint_position':joints,'state.joint_velocity':joints,'state.ee_position':3,
                    'state.gripper':1,'action.delta_position':3,'action.gripper':1}
        for key,length in expected.items():
            if key not in data or any(len(values) != length for values in data[key]):
                raise ValueError(f'upstream modality mismatch: {key}')
        language = data['language.annotation.human.task_description'].tolist()
        if any(text not in episode['tasks'] for text in language):
            raise ValueError('upstream language mapping mismatch')
        config = RELIC_CONFIG if decode_video else {key: value for key, value in RELIC_CONFIG.items() if key != 'video'}
        for frame_index in sorted({0, max(0, len(data)-16), len(data)-1}):
            sample = extract_step_data(data, frame_index, config, tag, allow_padding=True)
            indices = [min(frame_index + offset, len(data)-1) for offset in range(16)]
            for key in ('delta_position', 'gripper'):
                expected_actions = np.stack(data['action.'+key].iloc[indices].tolist()).astype(np.float32)
                if sample.actions[key].shape != expected_actions.shape or not np.array_equal(sample.actions[key], expected_actions):
                    raise ValueError('upstream action chunk alignment mismatch')
            normalized_state, normalized_action = processor.apply(sample.states, sample.actions, tag.value)
            processor.use_relative_action = False
            unchanged_action = processor.apply_action(sample.actions, tag.value, state=sample.states)
            processor.use_relative_action = True
            if any(not np.array_equal(normalized_action[key], unchanged_action[key]) for key in normalized_action):
                raise ValueError('world-frame actions received a second relative transformation')
            if any(not np.isfinite(values).all() for group in (normalized_state, normalized_action) for values in group.values()):
                raise ValueError('upstream normalization produced nonfinite values')
            if sample.text != language[frame_index]:
                raise ValueError('upstream sample instruction mismatch')
            if decode_video:
                image = sample.images['overhead'][0]
                if image.dtype != np.uint8 or list(image.shape) != info['features']['observation.images.overhead']['shape']:
                    raise ValueError('upstream decoded image mismatch')
            samples += 1
        frames += len(data)
    return {'groot_source_commit':GROOT_COMMIT,'upstream_loader':'LeRobotEpisodeLoader',
            'episodes':len(loader),'frames':frames,'video_decoding_executed':decode_video,
            'action_chunk_samples_verified':samples,'action_horizon':16,
            'upstream_normalization_verified':True,
            'world_action_semantics_verified':True,
            'training_executed':False,'inference_executed':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--decode-video',action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate_with_upstream(args.source,args.dataset,args.decode_video),indent=2))


if __name__ == '__main__':
    main()
