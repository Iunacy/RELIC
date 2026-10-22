#!/usr/bin/env python3
from collections import defaultdict,Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision as r
import revision_learning as rl
import pipeline as p


def load(path):return json.loads(path.read_text())


def interval(value,scale=1):
    return f"{value['mean']*scale:.6g} (95% CI {value['ci95'][0]*scale:.6g}–{value['ci95'][1]*scale:.6g})"


def main():
    out=r.DEFAULT;destination=r.ROOT/'adaptive/revision'
    primary=load(out/'comparison.json');robust=load(out/'analytical_robustness.json');teachers=load(out/'teacher-campaign/summary.json')
    for source,target in [
        ('comparison.json','comparison.json'),('analytical_robustness.json','analytical_robustness.json'),
        ('analytical_robustness_protocol.json','analytical_robustness_protocol.json'),('analytical_robustness_tuning.json','analytical_robustness_tuning.json'),
        ('convergence_comparison.json','convergence_comparison.json'),('convergence_replication_protocol.json','convergence_replication_protocol.json'),
        ('geometry_audit.json','geometry_audit.json'),('qualification.json','qualification.json'),
        ('teacher-campaign/summary.json','teacher_summary.json'),('prepared/timing.json','preparation_timing.json'),
        ('evaluation/timing.json','evaluation_timing.json'),('repeatability.json','repeatability.json'),('historical_replay.json','historical_replay.json'),
        ('demo_feedback.json','demo_feedback.json'),('comparison.png','comparison.png')]:
        shutil.copy2(out/source,destination/target)
    value=load(out/'protocol.json');compact={k:v for k,v in value.items() if k!='specs'}
    compact.update(full_protocol_sha256=p.digest(out/'protocol.json'),spec_count=len(value['specs']),
                   embodied_training_trajectories=128,embodied_evaluation_trajectories=80)
    p.atomic_json(destination/'split_protocol.json',compact)
    manifest=load(out/'teacher-campaign/teacher_manifest.json')
    attempts=[a for m in manifest for a in m['attempts']]
    outcomes=Counter()
    for file in out.glob('**/metadata.json'):outcomes[load(file)['outcome']]+=1
    p.atomic_json(destination/'run_inventory.json',dict(native_episode_directories=sum(outcomes.values()),outcomes=outcomes,
        main_teacher_scenarios=len(manifest),main_teacher_native_runs=len(manifest)+len(attempts)+sum(a['accepted'] for a in attempts),
        main_teacher_accepted=sum(a['accepted'] for a in attempts),main_teacher_attempts=len(attempts),
        main_teacher_rejections=[dict(episode=m['episode'],category=m['category'],severity=m['severity'],attempt=a['attempt'],reason=a['reason']) for m in manifest for a in m['attempts'] if not a['accepted']],
        generator_sha256=p.digest(out/'generator-pinned'),fixture_model_sha256=p.digest(destination/'models/shared_full.txt'),
        repository_head_at_export=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        native_compiled_base_commit=load(next((out/'teacher-campaign/teachers').glob('*/baseline/metadata.json')))['source_commit'],
        interrupted_preparation='serial preprocessing was stopped and resumed with four readers; existing arrays verified identical; serial log retained',
        bulk_run_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
        primary_evaluation_runs=8640,additional_robustness_evaluation_runs=640,
        process_failure_files=[str(f.relative_to(out)) for f in out.rglob('process_failure.json')]))
    banks={}
    for mode in ['matched','full']:
        rows=[]
        for spec in value['specs']:
            if not spec['partition'].endswith('_test') or spec['category'] not in value['primary_categories']:continue
            shared=load(out/'evaluation'/f'shared_{mode}'/spec['episode']/'metadata.json')
            expert=load(out/'evaluation'/f'per_{mode}'/spec['episode']/'metadata.json')
            rows.append((spec,shared,expert))
        banks[mode]={}
        for name,selection in [('primary',rows)]+[(f'dof:{d}',[row for row in rows if row[0]['dof']==d]) for d in [3,4]]:
            delta=defaultdict(list);success=defaultdict(list)
            for spec,a,b in selection:
                delta[spec['family']].append(a['metrics']['joint_rmse_rad']-b['metrics']['joint_rmse_rad'])
                success[spec['family']].append(float(a['task_success'])-float(b['task_success']))
            banks[mode][name]=dict(episodes=len(selection),shared_minus_expert_rmse_rad=rl.bootstrap([sum(v)/len(v) for _,v in sorted(delta.items())]),
                                   shared_minus_expert_success=rl.bootstrap([sum(v)/len(v) for _,v in sorted(success.items())]))
    p.atomic_json(destination/'model_bank_comparison.json',banks)
    methods=primary['methods'];shared=methods['shared_full']['groups']['primary'];strong=robust['groups']['primary']
    lines=['# Adaptive-v0 validation results','',
        'Learned correction improves substantially over conventional control, but the stronger analytical PI baseline has lower tracking RMSE. No learned-controller advantage over that baseline is claimed. The full-range PI study is exploratory: it was separately registered after one predeclared severe-case demo; all settings were selected using development data only. The original primary comparison is retained unchanged.','',
        '## Collection and split','',
        f"Collected {teachers['episodes']} teacher scenarios across {teachers['independent_base_families']} independent base families and {teachers['unique_trajectory_fingerprints']} embodied trajectory fingerprints: 512 episodes per embodiment, 128 per condition. Collection took {teachers['wall_s']:.2f} s ({teachers['episode_per_s']:.4f} teacher scenarios/s), including proposal replays, verification replays and concurrent development/testing work. There were 3,068 accepted proposals and four rejected proposals out of 3,072; all were retained. The campaign comprises 7,164 actual native executions. Category/severity rates and exact physical coverage are in teacher_summary.json.",
        '',
        'Eight of the 64 collected families were reserved for validation; 56 contributed fitting examples. Preparation produced 1,409,328 causal feature/target rows and rechecked every saved acceptance decision and verification hash. Four-reader preparation took 279.14 s. Serial preparation was interrupted for this performance revision; its log and prepared arrays were retained and checked for exact equality.',
        '',
        'The final evaluation has 40 new base families, 80 embodied trajectory fingerprints, both embodiments in every partition, and 640 conditions. Primary aggregates use 560 conditions, excluding the duplicated restored_nominal condition. Thirteen full-cohort methods plus a four-joint-only common-joint transfer probe produced 8,640 executions; the extra PI comparison adds 640. Bootstrap intervals use complete base-family clusters, 4,000 replicates and a fixed seed. No final runs were incomplete; no continuous pairs were omitted. There are no cross-partition trajectory fingerprints or shared seeds. The nearest audited endpoint max-norm distance is 0.090419 rad (near-duplicate rule: below .05 rad and duration difference below .1 s).',
        '',
        'restored_nominal is exactly nominal physics after a fresh reset. All 128 teacher nominal/restored pairs have identical physics and metrics. It is not evidence of degradation-then-recovery. Historical recovery-labelled records and the original 64-episode cohort remain unchanged as development evidence.',
        '', '## Closed-loop primary results','']
    for name in ['conventional','analytical','gravity','integral','shared_matched','per_matched','shared_full','per_full']:
        row=methods[name]['groups']['primary']
        lines.append(f"- {name}: RMSE {interval(row['joint_rmse_rad'])} rad; success {row['successes']}/{row['episodes']}; final Cartesian error {interval(row['final_cartesian_m'],1000)} mm.")
    lines+= [f"- full-range PI (exploratory): RMSE {interval(strong['joint_rmse_rad'])} rad; success 560/560; final Cartesian error {interval(strong['final_cartesian_m'],1000)} mm.",'',
        f"The paired PI-minus-shared RMSE difference is {interval(strong['paired']['shared_full']['rmse_delta_rad'])} rad. Both succeed on every primary condition. PI also has lower mean tracking error than the full expert bank. The learned model's slightly smaller final Cartesian error does not establish an overall learned advantage. Peak and command-end Cartesian errors, per-condition outcomes and paired intervals are retained in comparison.json and analytical_robustness.json.",
        '', '## Generalization and model comparisons','']
    for group in ['partition:combination_test','partition:severity_test']:
        lines.append(f"- {group}: conventional/shared/full-range-PI RMSE {methods['conventional']['groups'][group]['joint_rmse_rad']['mean']:.6g}/{methods['shared_full']['groups'][group]['joint_rmse_rad']['mean']:.6g}/{robust['groups'][group]['joint_rmse_rad']['mean']:.6g} rad; successes {methods['conventional']['groups'][group]['successes']}/160, 160/160, 160/160. Partition aggregates include the nominal duplicate; their full intervals are in the JSON reports.")
    lines+=['',
        'Matched training uses 60,000 rows per embodiment, 960 shared updates versus 480 per expert, and exactly the same actual minibatch exposures. The bank has 3,266 parameters versus 1,633 shared. Full-data training uses the same union of fitting examples for 24 epochs (14,592 shared updates and 14,592 total expert updates). These are separate comparisons. model_bank_comparison.json contains directly paired shared-minus-bank intervals, including separate embodiment results. No zero-shot transfer claim follows from shared training on both embodiments.','']
    for mode in ['matched','full']:
        result=banks[mode]['primary']
        lines.append(f"- {mode}: shared-minus-bank tracking RMSE {interval(result['shared_minus_expert_rmse_rad'])} rad; success-rate difference {interval(result['shared_minus_expert_success'])}.")
    lines+=['', 'Learning curves hold eight validation families fixed. Fitting-family counts are 7, 14, 28 and 56; total model-selection family exposure is 15, 22, 36 and 64.','']
    for name,count in [('curve_8',7),('curve_16',14),('curve_32',28),('shared_full',56)]:
        row=methods[name]['groups']['primary'];lines.append(f"- {count} fitting families: RMSE {interval(row['joint_rmse_rad'])} rad; success {row['successes']}/560.")
    transfer=methods['common_transfer']['groups']['primary']
    lines+=['',f"The explicit 3→4 probe predicts only joints 0–2 and leaves the wrist conventional. Common-joint RMSE is {interval(transfer['common_joints_rmse_rad'])} rad; full-arm success is 0/280 primary conditions (0/320 including restored_nominal). This is a transfer failure. The reverse direction was not tested.",'',
        'Zero and copy-previous execution traces are byte-identical to conventional control on every condition. For the full shared model, normalized validation MSE is 0.000052 versus zero 0.054658 and copy-previous 0.014154. All raw rad² MSEs, per-embodiment comparisons, model hashes, fitting counts, curves and training times are in training_results.json. Offline MSE is not used as a claim of closed-loop superiority.','',
        '## ILC convergence and safety','',
        'The original two-family convergence probe is retained. The expanded replication has 32 training families and 64 paired conditions per lead setting, eight fixed iterations, no evaluation labels and no use of extra iterations in model fitting. At iteration eight, zero-lead mean proposed RMSE is 0.00127717 rad and 40-ms lead is 0.00110987 rad. The paired lead-minus-zero difference is -0.000167297 rad (95% CI -0.000293910 to -0.0000711579). All 128 iteration-eight runs complete successfully. At that iteration, zero lead rejects six proposals and lead rejects ten; reasons include final-error regression and insufficient improvement. Proposal magnitude saturation is approximately 0.99%/0.96% and rate saturation 1.21%/1.17%. Every iteration, rejection and saturation count is retained in convergence_comparison.json. Accepted-best monotonicity follows the gate and is not a convergence proof.','',
        'All primary and extra-PI evaluation runs complete; no fallback, collision or invalid-command episodes were observed. Qualification includes 16 injected invalid-output runs at nonzero correction, all withdrawing to zero through validated rate-limited commands. Unit tests cover both correction signs, bad dimensions/nonfinite values, infeasible withdrawal and the distinct stop reason. Unsafe observed motion uses common stopping rules for every controller. These are sampled collision checks, not continuous guarantees.','',
        '## Tests, costs and preservation','',
        'The initial 34-test suite passed. The final release suite passes 35/35 (6.70 s); five adaptive-related checks pass under AddressSanitizer/UBSan (32.24 s). Eighty final controller executions reproduce byte-for-byte, and 16 historical conventional executions reproduce byte-for-byte with unchanged metrics. All 17 protected historical evidence/fixture files and the Ruckig status/diff hashes are unchanged. No existing test was removed or tolerance weakened.','',
        'Nine CPU model fits take approximately 147 s in total, excluding data preparation/import overhead. Inference cost and total execution wall time are measured in comparison.json/evaluation_timing.json; these are loaded-workstation measurements, not an isolated latency guarantee. Bulk datasets, builds and run directories remain ignored; only small models and reports are versioned. Native executable identity and the compiled dirty-base commit are recorded in run_inventory.json, rather than presenting the pinned executable as a later clean commit.','',
        '## Reproduction and limitations','',
        'Full campaign: `bash adaptive/revise.sh`. Quick actual-motion demonstration after building: `python adaptive/revision_demo.py --fixtures`. The model is a small committed text-weight MLP; the inference demo needs only the native executable and Python standard library. It uses the predeclared first severity-test family, four-joint combined-fault condition.','',
        'Exact commands and source changes are recorded in COMMANDS.md and the Git history. The run directory retains all proposal/repeat traces, rejected outcomes, the two-epoch integration probe (including its NonConvergence outcomes), the interrupted serial preparation log, primary comparisons and exploratory PI tuning. No unsuccessful run was silently removed.','',
        'Limitations: two synthetic arms, one training seed, finite family sample, no hardware, no continuous collision guarantee, no true recovery experiment, and failed one-direction common-joint transfer. Family intervals do not include training-seed uncertainty or multiplicity correction. Some numerical degradation ranges overlap even though the severity factor is held out; the combination test holds out parameter values, not new fault mechanisms. The research harness generates/executes quintic joint trajectories independently of production execute_cartesian_motion and uses its own long-form recorder rather than TraceWriter; production IK/planner selection and dwell semantics are not validated by these results. This recorder integration remains unfinished and is explicitly documented in PROTOCOL.md.','',
        'Recommended next action: review the stronger analytical PI result and the failed transfer before increasing model complexity or collecting more data. Keep the production controller unchanged. Any production integration or changed learning target should receive architectural review; the current results do not justify promoting the learned controller over the strongest tested analytical feedback. No Isaac Lab, Cosmos, GR00T or NVIDIA/Nebius integration files were changed.','']
    (destination/'RESULTS.md').write_text('\n'.join(lines))
    print('exported results and compact reproducibility artifacts')

if __name__=='__main__':main()
