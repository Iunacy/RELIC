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
    if run.get('initialization_protocol') == 'viper.matched_target.v1':
        keys += ('initialization_protocol', 'target_initialization_donor', 'trainable_selection',
                 'optimizer', 'normalizer_source', 'demonstration_budget', 'inference_config', 'safety_config')
    if not run.get('dataset_digest'):
        raise ValueError('validated adaptation dataset required')
    payload = {key:run[key] for key in keys}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_matched(baseline: dict, transfer: dict) -> str:
    if baseline['condition'] != 'b_only' or transfer['condition'] != 'a_to_b':
        raise ValueError('invalid transfer conditions')
    modern = baseline.get('initialization_protocol') == 'viper.matched_target.v1'
    if modern:
        if (transfer.get('initialization_protocol') != baseline['initialization_protocol'] or
                baseline.get('initialization_shared_weights') != 'pretrained' or
                transfer.get('initialization_shared_weights') != 'robot_a' or
                baseline['base_model'] == transfer['base_model']):
            raise ValueError('unmatched target initialization provenance')
    elif baseline['base_model'] != BASE_MODEL or transfer['base_model'] == BASE_MODEL:
        raise ValueError('baseline and transfer initializations must differ')
    if baseline['embodiment'] != 'franka_panda' or transfer['embodiment'] != baseline['embodiment']:
        raise ValueError('adaptation embodiment mismatch')
    signature = adaptation_signature(baseline)
    if signature != adaptation_signature(transfer):
        raise ValueError('unmatched adaptation data, budget, or evaluation protocol')
    return signature


def prepare(source: Path, dataset_a: Path, dataset_b: Path, output: Path, budget: Budget = Budget(), pretrained: Path | None = None) -> dict:
    verify_source(source)
    evaluations = [asdict(case) for case in plan('evaluation',50)]
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
    checkpoint_a = Path(output).resolve()/'a'
    for condition,robot,path,base,key in (
            ('a','openmanipulator_x',dataset_a,BASE_MODEL,'a'),
            ('b_only','franka_panda',dataset_b,BASE_MODEL,'b'),
            ('a_to_b','franka_panda',dataset_b,str(checkpoint_a),'b')):
        initialization = Path(output).resolve()/'initialization'/condition
        training_base = base if condition == 'a' else str(initialization)
        arguments = training_arguments(path,training_base,Path(output)/condition,budget)
        runs.append({'condition':condition,'embodiment':robot,'base_model':training_base,
                     'dataset_digest':datasets[key]['digest'],'budget':asdict(budget),
                     'evaluation_scenarios':evaluations,'success_definition':'relic.cube_bin.v2',
                     'action_horizon':16,'execution_horizon':16,
                     'arguments':arguments,'command':[sys.executable,str(Path(source).resolve()/'gr00t/experiment/launch_finetune.py'),*arguments],
                     'initialization_commands': ([[sys.executable,'-m','embodiment.policy',
                         '--prepare-training-checkpoint',str(checkpoint_a)],
                         [sys.executable, '-m', 'embodiment.transfer', '--source', str(checkpoint_a),
                          '--pretrained', str(pretrained.resolve()) if pretrained else '<local-pretrained-checkpoint>',
                          '--dataset', str(Path(dataset_b).resolve()), '--upstream', str(Path(source).resolve()),
                          '--output', str(initialization), '--condition', 'E2' if condition == 'b_only' else 'E3']]
                         if condition != 'a' else []),
                     'checkpoint':str(Path(output).resolve()/condition),
                     'training_executed':False,'checkpoint_save_load_verified':False})
    for run in runs[1:]:
        run.update(initialization_protocol='viper.matched_target.v1',
                   target_initialization_donor=str(pretrained.resolve()) if pretrained else BASE_MODEL,
                   initialization_shared_weights='pretrained' if run['condition'] == 'b_only' else 'robot_a',
                   trainable_selection=['action_head.state_encoder.*', 'action_head.action_encoder.*',
                       'action_head.action_decoder.*', 'action_head.model.*', 'action_head.position_embedding.*',
                       'action_head.vlln.*', 'action_head.vl_self_attention.*'],
                   optimizer='adamw_torch', normalizer_source=datasets['b']['digest'],
                   demonstration_budget=datasets['b'].get('episodes'),
                   inference_config={'prediction_horizon':16, 'execution_horizon':16},
                   safety_config='frozen robot-specific SafetyConfig; record actual digest at evaluation',
                   additional_source_budget=asdict(budget) if run['condition'] == 'a_to_b' else None,
                   actual_cost=None, trainable_names_runtime_verified=False)
    if pretrained is None:
        blockers.append('local pinned pretrained checkpoint required for matched E2/E3 target projector initialization')
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
    parser.add_argument('--pretrained-checkpoint',type=Path)
    parser.add_argument('--dataset-a',type=Path,required=True)
    parser.add_argument('--dataset-b',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=12)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--gradient-accumulation',type=int,default=1)
    parser.add_argument('--save-steps',type=int,default=500)
    args = parser.parse_args()
    budget = Budget(args.steps,args.batch_size,args.learning_rate,args.gradient_accumulation,42,args.save_steps)
    result = prepare(args.source,args.dataset_a,args.dataset_b,args.run_root,budget,args.pretrained_checkpoint)
    for run in result['runs']:
        validate_upstream_arguments(args.source,run['arguments'])
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output),'blockers':result['blockers']},indent=2))


if __name__ == '__main__':
    main()
