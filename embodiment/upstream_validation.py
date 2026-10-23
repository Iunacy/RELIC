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
    changes = subprocess.check_output(['git','-C',str(source),'diff','HEAD','--','gr00t'],text=True)
    if changes:
        raise ValueError('GR00T runtime source has local modifications')


def validate_with_upstream(source: Path, dataset: Path, decode_video: bool = False) -> dict:
    verify_source(source)
    sys.path.insert(0,str(Path(source).resolve()))
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from .groot_modalities import RELIC_CONFIG
    info = read_json(Path(dataset)/'meta/info.json')
    loader = LeRobotEpisodeLoader(dataset,RELIC_CONFIG)
    joints = DOF[info['robot_type']]+2
    frames = 0
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
        frames += len(data)
    return {'groot_source_commit':GROOT_COMMIT,'upstream_loader':'LeRobotEpisodeLoader',
            'episodes':len(loader),'frames':frames,'video_decoding_executed':decode_video,
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
