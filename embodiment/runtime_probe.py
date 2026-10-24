"""Import the actual pinned training/inference stack with network and GPU execution disabled."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from .upstream_validation import verify_source


def probe(source):
    verify_source(source)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(Path(source).resolve()))
    result = {'schema': 'viper.training_import_probe.v1', 'modules': {}, 'errors': [],
              'weights_loaded': False, 'optimizer_executed': False, 'inference_executed': False}
    for name in ('gr00t.experiment.experiment', 'gr00t.policy.gr00t_policy', 'torchcodec.decoders', 'deepspeed'):
        try:
            importlib.import_module(name)
            result['modules'][name] = 'IMPORTED'
        except Exception as error:
            result['modules'][name] = 'BLOCKED'
            result['errors'].append(f'{name}: {type(error).__name__}: {error}')
    result['ready'] = not result['errors']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.source)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
