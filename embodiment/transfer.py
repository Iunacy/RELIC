"""Prepare matched B initialization from real local N1.7 safetensors, offline."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from .checkpoint_compatibility import validate_processor
from .collection import atomic_json
from .contract import manifest
from .dataset import GROOT_COMMIT, read_json, validate_export
from .policy import checkpoint_digest, validate_checkpoint
from .training import dataset_digest
from .upstream_validation import verify_source

PROJECTORS = ('action_head.state_encoder.', 'action_head.action_encoder.', 'action_head.action_decoder.')
OPERATIONAL_CONFIG = {'_name_or_path', 'transformers_version', 'torch_dtype', 'dtype',
    'tune_llm', 'tune_visual', 'tune_projector', 'tune_diffusion_model', 'tune_vlln',
    'backbone_trainable_params_fp32', 'load_bf16', 'state_dropout_prob'}


def tensor_catalog(checkpoint):
    from safetensors import safe_open
    checked = validate_checkpoint(checkpoint, require_processor=False)
    catalog = {}
    for filename in checked['weight_paths']:
        with safe_open(Path(checkpoint)/filename, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in catalog:
                    raise ValueError('duplicate tensor across checkpoint shards')
                view = stream.get_slice(key)
                catalog[key] = {'file': filename, 'shape': list(view.get_shape()), 'dtype': view.get_dtype()}
    return catalog


def effective_configuration(config, upstream):
    verify_source(upstream)
    sys.path.insert(0, str(Path(upstream).resolve()))
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    return Gr00tN1d7Config(**config).to_filtered_dict(exclude_augment=True)


def validate_architecture(source, base, upstream=None):
    configs = [read_json(Path(path)/'config.json') for path in (source, base)]
    if any(str(config.get('model_type', '')).lower().replace('_', '') != 'gr00tn1d7' for config in configs):
        raise ValueError('transfer requires exact N1.7 model architecture')
    configs = [effective_configuration(config, upstream) for config in configs] if upstream else configs
    for config in configs: config.setdefault('state_history_length', 1)
    architecture_configs = [{key: value for key, value in config.items() if key not in OPERATIONAL_CONFIG}
                            for config in configs]
    if architecture_configs[0] != architecture_configs[1]:
        raise ValueError('source and pretrained architecture configurations differ')
    config = configs[0]
    if config.get('max_state_dim', 0) < 22 or config.get('max_action_dim', 0) < 4 or config.get('state_history_length') != 1:
        raise ValueError('architecture cannot represent the explicit B state and shared action')
    source_catalog, base_catalog = tensor_catalog(source), tensor_catalog(base)
    floating = {'F16', 'BF16', 'F32'}
    if set(source_catalog) != set(base_catalog) or any(
            value['shape'] != base_catalog[key]['shape'] or
            (value['dtype'] != base_catalog[key]['dtype'] and not {value['dtype'], base_catalog[key]['dtype']} <= floating)
            for key, value in source_catalog.items()):
        raise ValueError('missing, unexpected or incompatible source tensors')
    expected = {prefix+suffix for prefix in PROJECTORS for suffix in (
        ('layer1.W', 'layer1.b', 'layer2.W', 'layer2.b') if 'action_encoder' not in prefix else
        ('W1.W', 'W1.b', 'W2.W', 'W2.b', 'W3.W', 'W3.b'))}
    actual = {key for key in source_catalog if key.startswith(PROJECTORS)}
    if actual != expected:
        raise ValueError('unrecognized embodiment projector tensor layout')
    categories = config.get('max_num_embodiments', 0)
    if categories <= 10 or any(source_catalog[key]['shape'][0] != categories for key in expected):
        raise ValueError('projector category count differs from architecture')
    state = source_catalog['action_head.state_encoder.layer1.W']['shape']
    if len(state) != 3 or state[1] != config['max_state_dim']:
        raise ValueError('state projector input axis differs from upstream layout')
    return source_catalog, base_catalog


def target_processor(source, dataset, upstream):
    verify_source(upstream)
    report = validate_export(dataset)
    if report['embodiment'] != 'franka_panda' or report['split'] != 'train':
        raise ValueError('B initialization requires a verified Panda training dataset')
    validate_processor(source, 'openmanipulator_x')
    root = Path(validate_checkpoint(source)['processor'])
    configuration = deepcopy(read_json(root/'processor_config.json'))
    ids = read_json(root/'embodiment_id.json')
    if type(ids.get('new_embodiment')) is not int or ids['new_embodiment'] != 10:
        raise ValueError('pinned upstream NEW_EMBODIMENT projector index must be 10')
    kwargs = configuration['processor_kwargs']
    if kwargs.get('apply_sincos_state_encoding', False):
        raise ValueError('sincos encoding is outside the frozen joint-state contract')
    if kwargs.get('max_state_dim', 0) < 22 or kwargs.get('max_action_dim', 0) < 4:
        raise ValueError('processor padding capacity cannot represent B')
    sys.path.insert(0, str(Path(upstream).resolve()))
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from .groot_modalities import RELIC_CONFIG
    statistics = {'new_embodiment': LeRobotEpisodeLoader(Path(dataset), RELIC_CONFIG).get_dataset_statistics()}
    return configuration, statistics, ids


def prepare(source, pretrained, dataset, upstream, output, condition='E3'):
    if condition not in ('E2', 'E3', 'E4'):
        raise ValueError('supported initializations are E2, E3 and unadapted E4')
    source, pretrained, dataset, upstream, output = map(lambda value: Path(value).resolve(),
                                                      (source, pretrained, dataset, upstream, output))
    if output.exists():
        raise ValueError('initialization output already exists; verify its manifest instead of overwriting')
    before = checkpoint_digest(source)
    data_before = dataset_digest(dataset)
    source_catalog, base_catalog = validate_architecture(source, pretrained, upstream)
    configuration, statistics, ids = target_processor(source, dataset, upstream)
    base_files = validate_checkpoint(pretrained, require_processor=False)['weight_paths']
    base_digest = hashlib.sha256()
    for name in ['config.json', *base_files]:
        with (pretrained/name).open('rb') as stream:
            while chunk := stream.read(1024*1024): base_digest.update(chunk)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.viper-init-', dir=output.parent))
    try:
        donor = pretrained if condition == 'E2' else source
        donor_manifest = validate_checkpoint(donor, require_processor=False)
        from safetensors import safe_open
        from safetensors.torch import save_file
        reset = []
        for filename in donor_manifest['weight_paths']:
            tensors = {}
            with safe_open(donor/filename, framework='pt', device='cpu') as saved:
                for key in saved.keys():
                    tensor = saved.get_tensor(key)
                    if key.startswith(PROJECTORS):
                        with safe_open(pretrained/base_catalog[key]['file'], framework='pt', device='cpu') as base:
                            tensor = tensor.clone()
                            tensor[10].copy_(base.get_tensor(key)[10])
                        reset.append(key)
                    tensors[key] = tensor
            save_file(tensors, temporary/filename, metadata={'format': 'pt'})
            del tensors
        for name in ('config.json', 'model.safetensors.index.json'):
            if (donor/name).exists(): shutil.copyfile(donor/name, temporary/name)
        for name, value in (('processor_config.json', configuration), ('statistics.json', statistics), ('embodiment_id.json', ids)):
            atomic_json(temporary/name, value)
        validate_processor(temporary, 'franka_panda')
        validate_checkpoint(temporary)
        if checkpoint_digest(source) != before:
            raise ValueError('source checkpoint changed during conversion')
        if dataset_digest(dataset) != data_before:
            raise ValueError('target dataset changed during statistics conversion')
        result = {'schema': 'viper.transfer_initialization.v1', 'condition': condition,
                  'source_checkpoint_sha256': before, 'pretrained_weights_sha256': base_digest.hexdigest(),
                  'target_dataset_sha256': data_before, 'groot_commit': GROOT_COMMIT,
                  'target_contract': manifest('franka_panda'), 'target_projector_index': 10,
                  'target_projectors': {key: 'pretrained slot 10; identical E2/E3 initialization' for key in sorted(reset)},
                  'inherited_tensors': {key: {'donor': 'pretrained' if condition == 'E2' else 'robot_a',
                      **value} for key, value in source_catalog.items() if key not in reset},
                  'other_projector_slots': 'retained from donor; inactive for NEW_EMBODIMENT',
                  'normalizer': 'recomputed by pinned upstream reader from B training data only',
                  'checkpoint_sha256': checkpoint_digest(temporary),
                  'upstream_training_load_verified': False, 'upstream_inference_verified': False,
                  'adapter_status': 'initialized from pretrained; no target optimizer steps',
                  'runtime_gate': 'strict upstream load, B optimizer steps, save/reload, inference and closed-loop required'}
        atomic_json(temporary/'initialization.json', result)
        temporary.rename(output)
        return result
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pretrained', 'dataset', 'upstream', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=['E2', 'E3', 'E4'], default='E3')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.pretrained, args.dataset, args.upstream, args.output, args.condition), indent=2))


if __name__ == '__main__':
    main()
