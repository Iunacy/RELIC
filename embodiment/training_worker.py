"""Instrument the pinned upstream single-GPU trainer without changing its optimizer."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
from .collection import atomic_json
from .upstream_validation import verify_source


def parameter_inventory(model):
    return {name: {'shape': list(parameter.shape), 'elements': parameter.numel()}
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def run(source, metrics, arguments):
    verify_source(source)
    sys.path.insert(0, str(Path(source).resolve()))
    import torch
    from transformers import TrainerCallback
    import gr00t.experiment.experiment as experiment
    result = {'schema': 'viper.training_resources.v1', 'training_process_completed': False,
              'trainable_parameters': {}, 'peak_allocated_gpu_bytes': None, 'error': None}
    started = time.monotonic()
    class Inventory(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            result['trainable_parameters'] = parameter_inventory(model)
            atomic_json(metrics, result)
    original = experiment.Gr00tTrainer
    class AuditedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(Inventory())
    experiment.Gr00tTrainer = AuditedTrainer
    previous_argv = sys.argv
    try:
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        sys.argv = [str(Path(source)/'gr00t/experiment/launch_finetune.py'), *arguments]
        runpy.run_path(sys.argv[0], run_name='__main__')
        result['training_process_completed'] = True
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        sys.argv = previous_argv
        experiment.Gr00tTrainer = original
        result['wall_seconds'] = time.monotonic()-started
        if torch.cuda.is_available(): result['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
        atomic_json(metrics, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    args, training_args = parser.parse_known_args()
    run(args.source, args.metrics, training_args)
