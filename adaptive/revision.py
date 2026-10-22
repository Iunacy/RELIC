#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import concurrent.futures as futures
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
import learning as l
import pipeline as p

CATEGORIES = ['nominal', 'gain', 'force', 'friction', 'mass', 'payload', 'combined', 'restored_nominal']
ROOT = p.ROOT
DEFAULT = ROOT/'adaptive/runs/revision-20261022'
BOUNDS = ROOT/'adaptive/revision/bounds.txt'


def rank(value):
    return hashlib.sha256(f'relic-revision-20261022:{value}'.encode()).hexdigest()


def protocol():
    pool = sorted(range(10000), key=rank)
    groups = [('train', pool[:64]), ('development', pool[64:72])]
    groups += [(name, pool[72+10*i:82+10*i]) for i, name in enumerate(['trajectory_test', 'combination_test', 'severity_test', 'transfer_test'])]
    specs = []
    for partition, families in groups:
        for family in families:
            for dof in (3, 4):
                for category in CATEGORIES:
                    family_id = 9202610220000+family
                    seed = int(rank(f'{family}:{category}')[:12], 16)
                    severity = .85+(seed%13)/100 if partition == 'severity_test' else .25+(seed%43)/100
                    if partition == 'combination_test' and category == 'combined':
                        severity = .78
                    index = len(specs)
                    specs.append(dict(index=index, family=family_id, seed=seed, dof=dof, category=category, severity=severity,
                                      partition=partition, episode=f'r{index:06d}'))
    return dict(version='relic.validation.v1', seed=20261022, specs=specs, categories=CATEGORIES,
                family_order={k: [9202610220000+v for v in fs] for k, fs in groups},
                acceptance='learning.acceptance: complete; Success/NonConvergence; no fallback/contact; RMSE < .995 best; final <= max(best+.00025,.0005); identical replay hash',
                primary_categories=CATEGORIES[:-1], restored_nominal='fresh reset with exactly nominal physics; reproducibility control, not degradation-then-recovery',
                bootstrap=dict(seed=20261022, replications=4000, unit='base trajectory family; both embodiments and all conditions retained', confidence=.95),
                teacher_attempts=3, convergence_attempts=8, lag_lead_ticks=[0,20],
                learning_curve_family_counts=[8,16,32,64], epochs=24, stride=16,
                matched_rows_per_embodiment=60000, matched_updates_per_embodiment=480,
                safety='same native monitor and bounds for all methods; invalid residual latches validated rate-limited withdrawal; infeasible withdrawal terminates',
                inference_features='relic.local.v1 causal pre-state, past errors and applied correction; no future observations or randomized physics',
                transfer='per_3dof weights on joints 0..2 of 4dof; wrist conventional; common-joint RMSE separate from full-arm success',
                primary_metrics=['joint_rmse_rad','command_end_cartesian_m','final_cartesian_m','task_success'],
                incomplete_policy='report all failures; continuous paired deltas only if both complete, report omitted count; success counts every condition',
                analytical_grid=[dict(lead=lead,ki=ki,kp=kp) for lead in [.05,.1,.15] for ki,kp in [(0,0),(.5,.1),(1.,.2)]],
                tuning='development only; lowest mean RMSE among candidates completing every run without safety failures; tie by config order',
                bounds_sha256=p.digest(BOUNDS), bounds_origin='historical training-only calibration, frozen before revision; unchanged production gains')


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    path=out/'protocol.json'
    value=protocol()
    if path.exists():
        assert json.loads(path.read_text())==value, 'frozen protocol mismatch'
    else:
        p.atomic_json(path,value)
    import revision_geometry as geometry
    p.atomic_json(out/'geometry_audit.json',geometry.audit(value))
    source=out/'generator-pinned'
    executable=ROOT/'build-adaptive-validation/adaptive/relic_episode'
    if not source.exists():
        shutil.copy2(executable,source)
    return value


def run_task(task):
    out, m, method, model, control = task
    path=out/method/m['episode']
    if (path/'metadata.json').exists():
        meta=p.validate_episode(path)
        assert all(meta[key]==m[key] for key in ['seed','family','dof','category','severity','partition','episode']),'resume spec mismatch'
        assert meta['executable_sha256']==p.digest(out.parent/'generator-pinned'),'resume executable mismatch'
        assert meta['bounds_sha256']==p.digest(BOUNDS),'resume bounds mismatch'
        assert meta['inference_model_sha256']==(p.digest(Path(model)) if Path(model).is_file() else None),'resume model mismatch'
        if control is not None:
            assert meta['native_argv'][-1]==str(control),'resume control path mismatch'
            if 'control_configuration_sha256' in meta:assert meta['control_configuration_sha256']==p.digest(Path(control)),'resume control mismatch'
        return meta
    if path.exists():path.rename(path.with_name(path.name+f'-interrupted-{time.time_ns()}'))
    return l.run_native(out.parent/'generator-pinned',ROOT/'examples',path,m,
                        'mlp' if method.startswith(('shared','per_','curve')) else 'common_transfer' if method=='common_transfer' else method,
                        BOUNDS,model,control)


def execute(tasks, workers=6):
    records=[]
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(run_task,t):t for t in tasks}
        for job in futures.as_completed(jobs):
            try:
                records.append(job.result())
            except Exception as error:
                p.atomic_json(jobs[job][0]/f'failure-{jobs[job][2]}-{jobs[job][1]["episode"]}.json',dict(error=str(error)))
                raise
            if len(records)%32==0:
                print(f'{len(records)}/{len(tasks)} simulations',flush=True)
    return records


def config_file(out,name,lead,ki=0,kp=0,inject=-1):
    path=out/f'{name}.control'
    path.write_text(f'RELIC_CONTROL_V1 {lead} {ki} {kp} .05 {inject}\n')
    return path


def qualify(out):
    value=freeze(out)
    selected=value['specs'][:16]+value['specs'][64*16:64*16+16]
    began=time.perf_counter()
    results=execute([(out/'qualification',m,method,'none',None) for m in selected for method in ['conventional','gravity','integral','zero','copy_previous']])
    injected=config_file(out,'injected',.15,inject=300)
    results+=execute([(out/'qualification',dict(m,episode=m['episode']+'-injected'),'analytical','none',injected) for m in selected[:16]])
    injections=[r for r in results if r['episode'].endswith('injected')]
    assert all(r['fallbacks']>0 for r in injections)
    for r in injections:
        rows=l.read_rows(out/'qualification/analytical'/r['episode'])
        first=next(int(row['tick']) for row in rows if row['fallback']=='1')
        prior=[row for row in rows if int(row['tick'])==first-1]
        assert any(abs(float(row['correction_rad']))>1e-6 for row in prior), 'injection must occur at nonzero correction'
        for j in range(r['dof']):
            sequence=[float(row['correction_rad']) for row in rows if int(row['joint'])==j and int(row['tick'])>=first]
            assert all(abs(b)<=abs(a)+1e-12 for a,b in zip(sequence,sequence[1:]))
            assert abs(sequence[-1])<1e-12
    p.atomic_json(out/'qualification.json',dict(episodes=len(results),wall_s=time.perf_counter()-began,
                  outcomes=dict(Counter(r['outcome'] for r in results)),nonzero_injections=len(injections),protocol_sha256=p.digest(out/'protocol.json')))


def teacher_task(task):
    out,m,attempts,lead=task
    folder=out/'teachers'/m['episode']
    done=folder/'result.json'
    if done.exists():
        return json.loads(done.read_text())
    args=SimpleNamespace(output=out,exe=out.parent/'generator-pinned',examples=ROOT/'examples',bounds=BOUNDS,attempts=attempts)
    def run(path,controller,model='none'):
        if (path/'metadata.json').exists():
            meta=p.validate_episode(path)
            assert all(meta[k]==m[k] for k in ['episode','family','dof','category','severity','partition'])
            assert meta['executable_sha256']==p.digest(args.exe)
            if controller=='replay':assert meta['inference_model_sha256']==p.digest(model)
            return meta
        if path.exists():
            archived=path.with_name(path.name+f'-interrupted-{time.time_ns()}')
            path.rename(archived)
        return l.run_native(args.exe,args.examples,path,m,controller,BOUNDS,model)
    baseline=run(folder/'baseline','conventional')
    best,best_path=baseline,folder/'baseline'
    accepted,records=[],[]
    magnitude,rate=l.bound_values(BOUNDS)
    for attempt in range(attempts):
        seq,error=l.table_from_trace(best_path)
        shifted=[error[min(k+lead,len(error)-1)] for k in range(len(error))]
        rows=l.proposal(seq,shifted,BOUNDS,1. if attempt==0 else .7)
        replay=folder/f'proposal_{attempt}.txt'
        if replay.exists():
            expected=folder/f'proposal_{attempt}.resume.txt';l.write_table(expected,rows)
            assert p.digest(expected)==p.digest(replay),'resume proposal changed'
            expected.unlink()
        else:l.write_table(replay,rows)
        trial_path=folder/f'attempt_{attempt}'
        trial=run(trial_path,'replay',replay)
        passed,why=l.acceptance(trial,best)
        repeat_hash=None
        if passed:
            repeat=run(folder/f'repeat_{attempt}','replay',replay)
            repeat_hash=repeat['trace_sha256']
            assert repeat_hash==trial['trace_sha256'],'invalid nonrepeatable labels: stop for review'
            best,best_path=trial,trial_path;accepted.append(str(trial_path.resolve()))
        magnitude_hits=sum(abs(row[j])>=magnitude[j]-1e-12 for row in rows for j in range(m['dof']))
        rate_hits=sum(abs(rows[k][j]-rows[k-1][j])>=rate[j]-1e-12 for k in range(1,len(rows)) for j in range(m['dof']))
        records.append(dict(attempt=attempt,accepted=passed,reason=why,trial_rmse=trial['metrics']['joint_rmse_rad'],best_rmse=best['metrics']['joint_rmse_rad'],
                            category=m['category'],severity=m['severity'],path=str(trial_path.resolve()),trace_sha256=trial['trace_sha256'],repeat_sha256=repeat_hash,
                            magnitude_saturated_samples=magnitude_hits,rate_saturated_samples=rate_hits,total_joint_ticks=(len(rows)-1)*m['dof'],lead_ticks=lead,
                            proposal_sha256=p.digest(replay),outcome=trial['outcome'],fallbacks=trial['fallbacks']))
        p.atomic_json(folder/'attempts.json',records)
    result=dict(**{k:m[k] for k in ['episode','family','dof','partition','category','severity']},accepted=accepted,attempts=records,
                baseline_rmse=baseline['metrics']['joint_rmse_rad'],best_rmse=best['metrics']['joint_rmse_rad'])
    p.atomic_json(done,result)
    return result


def collect(out,qualification=False):
    value=freeze(out)
    assert (out/'qualification.json').exists(),'qualify before scaling'
    selected=[m for m in value['specs'] if m['partition']=='train']
    if qualification:
        selected=selected[:32]
    campaign=out/('teacher-qualification' if qualification else 'teacher-campaign')
    campaign.mkdir(exist_ok=True)
    began=time.perf_counter();results=[]
    with futures.ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(teacher_task,[(campaign,m,3,0) for m in selected]):
            results.append(result)
            p.atomic_json(campaign/'teacher_manifest.json',results)
            if len(results)%16==0:print(f'{len(results)}/{len(selected)} teacher episodes',flush=True)
    summarize_teachers(campaign,results,time.perf_counter()-began)


def summarize_teachers(campaign,results,wall):
    groups=defaultdict(lambda:dict(attempts=0,accepted=0))
    fingerprints=set();physics=defaultdict(list);nominal_pairs={}
    for r in results:
        base=json.loads((campaign/'teachers'/r['episode']/'baseline/metadata.json').read_text())
        fingerprints.add(p.trajectory_key(base))
        for key in ['gain_scale','force_scale','damping_scale','mass_scale','friction_Nm','payload_kg']:physics[key].append(base['physics'][key])
        if r['category'] in ['nominal','restored_nominal']:nominal_pairs.setdefault((r['family'],r['dof']),{})[r['category']]=base
        for a in r['attempts']:
            for key in [r['category'],f"severity_{int(r['severity']*10)/10:.1f}"]:
                groups[key]['attempts']+=1;groups[key]['accepted']+=int(a['accepted'])
    for item in groups.values():item['acceptance_rate']=item['accepted']/item['attempts']
    paired=0
    for pair in nominal_pairs.values():
        if len(pair)==2:
            assert pair['nominal']['physics']==pair['restored_nominal']['physics']
            assert pair['nominal']['metrics']==pair['restored_nominal']['metrics']
            paired+=1
    p.atomic_json(campaign/'summary.json',dict(nominal_restored_identical_pairs=paired,physical_coverage={key:dict(min=min(values),max=max(values),unique_values=len(set(values))) for key,values in physics.items()},episodes=len(results),independent_base_families=len({r['family'] for r in results}),
                  embodied_families=len({(r['family'],r['dof']) for r in results}),unique_trajectory_fingerprints=len(fingerprints),
                  categories=dict(Counter(r['category'] for r in results)),dofs=dict(Counter(r['dof'] for r in results)),wall_s=wall,
                  episode_per_s=len(results)/wall,acceptance=groups))


def convergence(out):
    value=freeze(out);selection=[m for m in value['specs'] if m['partition']=='train'][:32]
    for lead in value['lag_lead_ticks']:
        directory=out/f'convergence-lead-{lead}';directory.mkdir(exist_ok=True)
        began=time.perf_counter()
        with futures.ProcessPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(teacher_task,[(directory,m,8,lead) for m in selection]))
        p.atomic_json(directory/'teacher_manifest.json',results)
        summarize_teachers(directory,results,time.perf_counter()-began)
        p.atomic_json(directory/'convergence.json',[dict(iteration=i+1,mean_proposal_rmse=sum(r['attempts'][i]['trial_rmse'] for r in results)/len(results),
                      accepted=sum(r['attempts'][i]['accepted'] for r in results),rejected=sum(not r['attempts'][i]['accepted'] for r in results),
                      mean_best_rmse=sum(r['attempts'][i].get('best_rmse',r['attempts'][i]['trial_rmse']) for r in results)/len(results),
                      magnitude_saturated_samples=sum(r['attempts'][i].get('magnitude_saturated_samples',0) for r in results),
                      rate_saturated_samples=sum(r['attempts'][i].get('rate_saturated_samples',0) for r in results),
                      total_joint_ticks=sum(r['attempts'][i].get('total_joint_ticks',0) for r in results)) for i in range(8)])


def tune(out):
    value=freeze(out);selection=[m for m in value['specs'] if m['partition']=='development']
    records=[]
    for i,c in enumerate(value['analytical_grid']):
        method='gravity' if c['ki']==0 else 'integral'
        control=config_file(out,f'tune-{i}',c['lead'],c['ki'],c['kp'])
        directory=out/f'tuning-{i}'
        results=execute([(directory,m,method,'none',control) for m in selection])
        safe=all(r['completed_ticks']==r['planned_ticks'] and r['outcome'] in ['Success','NonConvergence'] and r['fallbacks']==0 for r in results)
        records.append(dict(index=i,method=method,config=c,safe=safe,mean_rmse=sum(r['metrics']['joint_rmse_rad'] for r in results)/len(results),control=str(control)))
        p.atomic_json(out/'tuning.json',records)
    winners={method:min([r for r in records if r['method']==method and r['safe']],key=lambda r:r['mean_rmse']) for method in ['gravity','integral']}
    p.atomic_json(out/'tuned.json',winners)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','qualify','teacher-qualify','collect','convergence','tune','prepare','train','evaluate','evaluate-baselines','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve()
    if args.command=='freeze':freeze(out)
    elif args.command=='qualify':qualify(out)
    elif args.command=='teacher-qualify':collect(out,True)
    elif args.command=='collect':collect(out)
    elif args.command=='convergence':convergence(out)
    elif args.command=='tune':tune(out)
    else:
        import revision_learning as rl
        if args.command=='evaluate-baselines':rl.evaluate(out,True)
        else:getattr(rl,args.command)(out)

if __name__=='__main__':main()
