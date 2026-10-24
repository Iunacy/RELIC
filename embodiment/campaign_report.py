"""Hash-verified comparisons from completed physical evaluation campaigns."""
import argparse
import json
from pathlib import Path
from .collection import atomic_json
from .dataset import file_digest, read_json
from .paired import compare, verify_evidence
from .reference_comparison import compare as compare_reference


def report(root, reference, observation_disturbance='baseline'):
    root, reference = Path(root), Path(reference)
    evaluations, paths = {}, {}
    suffix = '_'+observation_disturbance if observation_disturbance != 'baseline' else ''
    conditions = ('a', 'b_only', 'a_to_b') if not suffix else ('b_only', 'a_to_b')
    for condition in conditions:
        receipt = read_json(root/('evaluate_'+condition+suffix)/'completed.json')
        path = Path(receipt['report'])
        if receipt['status'] != 'COMPLETE' or file_digest(path) != receipt['report_sha256']:
            raise ValueError('evaluation receipt or report integrity failure')
        measured = read_json(path)
        verify_evidence(path.parent, measured)
        if measured['policy_service']['checkpoint_sha256'] != receipt['checkpoint_sha256']:
            raise ValueError('evaluation checkpoint differs from completion receipt')
        evaluations[condition], paths[condition] = measured, path
    paired = compare(evaluations['b_only'], evaluations['a_to_b'])
    result = {'schema': 'viper.campaign_comparison.v1', 'status': 'VERIFIED',
              'paired_target_adaptation': paired, 'raw_episode_hashes_verified': True,
              'classical_comparisons': {name: compare_reference(reference, path) for name, path in paths.items()} if not suffix else {},
              'observation_disturbance': observation_disturbance,
              'input_sha256': {name: file_digest(path) for name, path in paths.items()}}
    atomic_json(root/('comparison'+suffix+'.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    from .perception import KINDS as OBSERVATION_KINDS
    parser.add_argument('--observation-disturbance', choices=OBSERVATION_KINDS, default='baseline')
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.reference, args.observation_disturbance), indent=2))
