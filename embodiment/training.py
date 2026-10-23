import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from .dataset import GROOT_COMMIT, read_json, validate_export
from .scenarios import plan
from .upstream_validation import verify_source

BASE_MODEL = 'nvidia/GR00T-N1.7-3B'
BACKBONE = 'nvidia/Cosmos-Reason2-2B'
MODALITY_PATH = Path(__file__).resolve().with_name('groot_modalities.py')


@dataclass(frozen=True)
class Budget:
    steps: int = 2000
    batch_size: int = 12
    learning_rate: float = 1e-4
    gradient_accumulation: int = 1
    seed: int = 42
    save_steps: int = 500

    def __post_init__(self):
        for value in (self.steps,self.batch_size,self.gradient_accumulation,self.save_steps):
            if type(value) is not int or value <= 0:
                raise ValueError('training counts must be positive integers')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.save_steps > self.steps:
            raise ValueError('invalid training schedule')
        if type(self.seed) is not int or self.seed != 42:
            raise ValueError('the pinned upstream launcher uses the default seed 42')


def training_arguments(dataset: Path, base: str, output: Path, budget: Budget) -> list[str]:
    return ['--base-model-path',base,'--dataset-path',str(Path(dataset).resolve()),
            '--embodiment-tag','NEW_EMBODIMENT','--modality-config-path',str(MODALITY_PATH),
            '--num-gpus','1','--output-dir',str(Path(output).resolve()),
            '--max-steps',str(budget.steps),'--global-batch-size',str(budget.batch_size),
            '--learning-rate',str(budget.learning_rate),'--gradient-accumulation-steps',str(budget.gradient_accumulation),
            '--save-steps',str(budget.save_steps),'--save-total-limit','3','--dataloader-num-workers','2',
            '--tune-projector','--tune-diffusion-model','--no-tune-llm','--no-tune-visual']


def dataset_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob('*')):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            with file.open('rb') as stream:
                while chunk := stream.read(1024*1024):
                    digest.update(chunk)
    return digest.hexdigest()


def adaptation_signature(run: dict) -> str:
    keys = ('dataset_digest','budget','evaluation_scenarios','success_definition','action_horizon','execution_horizon')
    if not run.get('dataset_digest'):
        raise ValueError('validated adaptation dataset required')
    payload = {key:run[key] for key in keys}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_matched(baseline: dict, transfer: dict) -> str:
    if baseline['condition'] != 'b_only' or transfer['condition'] != 'a_to_b':
        raise ValueError('invalid transfer conditions')
    if baseline['base_model'] != BASE_MODEL or transfer['base_model'] == BASE_MODEL:
        raise ValueError('baseline and transfer initializations must differ')
    if baseline['embodiment'] != 'franka_panda' or transfer['embodiment'] != baseline['embodiment']:
        raise ValueError('adaptation embodiment mismatch')
    signature = adaptation_signature(baseline)
    if signature != adaptation_signature(transfer):
        raise ValueError('unmatched adaptation data, budget, or evaluation protocol')
    return signature


def prepare(source: Path, dataset_a: Path, dataset_b: Path, output: Path, budget: Budget = Budget()) -> dict:
    verify_source(source)
    evaluations = [asdict(case) for case in plan('evaluation',20)]
    blockers, datasets = [], {}
    for name,path,robot in (('a',dataset_a,'openmanipulator_x'),('b',dataset_b,'franka_panda')):
        try:
            report = validate_export(path)
            if report['embodiment'] != robot or report['split'] != 'train':
                raise ValueError('training embodiment or split mismatch')
            datasets[name] = {**report,'digest':dataset_digest(path)}
        except (OSError,ValueError,KeyError) as error:
            datasets[name] = {'digest':None}
            blockers.append(f'dataset_{name}: {type(error).__name__}: {error}')
    if all(datasets[name].get('episodes') for name in ('a','b')) and datasets['a']['episodes'] != datasets['b']['episodes']:
        blockers.append('scripted episode counts differ across embodiments')
    runs = []
    checkpoint_a = Path(output).resolve()/'a'/f'checkpoint-{budget.steps}'
    for condition,robot,path,base,key in (
            ('a','openmanipulator_x',dataset_a,BASE_MODEL,'a'),
            ('b_only','franka_panda',dataset_b,BASE_MODEL,'b'),
            ('a_to_b','franka_panda',dataset_b,str(checkpoint_a),'b')):
        arguments = training_arguments(path,base,Path(output)/condition,budget)
        runs.append({'condition':condition,'embodiment':robot,'base_model':base,
                     'dataset_digest':datasets[key]['digest'],'budget':asdict(budget),
                     'evaluation_scenarios':evaluations,'success_definition':'relic.cube_bin.v2',
                     'action_horizon':16,'execution_horizon':16,
                     'arguments':arguments,'command':[sys.executable,str(Path(source).resolve()/'gr00t/experiment/launch_finetune.py'),*arguments],
                     'checkpoint':str(Path(output).resolve()/condition/f'checkpoint-{budget.steps}'),
                     'training_executed':False,'checkpoint_save_load_verified':False})
    matched = None
    if datasets['b']['digest']:
        matched = validate_matched(runs[1],runs[2])
    access = {'model':BASE_MODEL,'backbone':BACKBONE,'weights_access_verified':False}
    blockers.extend(['model and gated backbone access must be verified on the training host',
                     'a paid GPU smoke test requires separate spending authorization',
                     'checkpoint save, reload and action inference have not been executed'])
    return {'schema':'relic.training_plan.v1','groot_commit':GROOT_COMMIT,'runs':runs,'datasets':datasets,
            'matched_b_adaptation_signature':matched,'model_access':access,'blockers':blockers,
            'paid_compute_go':False,'training_executed':False,
            'pretraining_overlap':{'pretrain_tag':'oxe_droid_relative_eef_relative_joint',
                                  'panda_posttrain_tags':['libero_sim','robocasa_panda_omron'],
                                  'exact_franka_pretraining_exposure':'not established; retain overlap caveat'},
            'action_semantics':'stored world-frame increments; no second relative-action transform',
            'adapter_training':'projector and diffusion model retrained for each embodiment; language and visual backbones frozen'}


def validate_upstream_arguments(source: Path, arguments: list[str]) -> dict:
    verify_source(source)
    sys.path.insert(0,str(Path(source).resolve()))
    import tyro
    from gr00t.configs.finetune_config import FinetuneConfig
    config = tyro.cli(FinetuneConfig,args=arguments)
    return asdict(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset-a',type=Path,required=True)
    parser.add_argument('--dataset-b',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = prepare(args.source,args.dataset_a,args.dataset_b,args.run_root)
    for run in result['runs']:
        validate_upstream_arguments(args.source,run['arguments'])
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output),'blockers':result['blockers']},indent=2))


if __name__ == '__main__':
    main()
