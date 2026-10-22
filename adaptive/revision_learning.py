#!/usr/bin/env python3
from collections import Counter, defaultdict
import concurrent.futures as futures
import hashlib
import json
import math
from pathlib import Path
import time
import learning as l
import pipeline as p
import revision as r


def prepare_one(task):
    import numpy as np
    out,directory,stride,m=task
    if not m['accepted']:return None
    baseline=out/'teacher-campaign/teachers'/m['episode']/'baseline'
    gate_best=json.loads((baseline/'metadata.json').read_text())
    for attempt in m['attempts']:
        trial_path=Path(attempt['path']);trial=json.loads((trial_path/'metadata.json').read_text())
        assert p.digest(trial_path/'telemetry.csv.gz')==trial['trace_sha256']==attempt['trace_sha256'],'trial trace identity'
        assert all(trial[key]==gate_best[key] for key in ['seed','family','dof','category','severity','initial_q_rad','goal_q_rad','physics','planned_ticks']),'teacher paired-condition mismatch'
        passed,why=l.acceptance(trial,gate_best)
        assert (passed,why)==(attempt['accepted'],attempt['reason']),'invalid teacher gate: stop for review'
        if passed:
            repeat_path=trial_path.with_name(f"repeat_{attempt['attempt']}")
            repeat=json.loads((repeat_path/'metadata.json').read_text())
            assert p.digest(repeat_path/'telemetry.csv.gz')==repeat['trace_sha256']==attempt['repeat_sha256']==trial['trace_sha256'],'invalid verification replay: stop for review'
            gate_best=trial
    best=Path(m['accepted'][-1]);meta=p.validate_episode(best)
    assert meta['partition']=='train' and all(meta[key]==m[key] for key in ['family','dof','category','severity','episode'])
    assert any(a['accepted'] and a['path']==str(best) and a['trace_sha256']==a['repeat_sha256']==meta['trace_sha256'] for a in m['attempts'])
    targets,_=l.table_from_trace(best)
    rows=[]
    for folder in [best.parent/'baseline']+[Path(path) for path in m['accepted']]:
        rows.extend(l.examples_from_trace(folder,stride=stride,targets=targets))
    x=np.asarray([row[0] for row in rows],dtype=np.float32)
    y=np.asarray([row[1] for row in rows],dtype=np.float32)
    j=np.asarray([row[2] for row in rows],dtype=np.int8)
    path=directory/f'{m["episode"]}.npz'
    if path.exists():
        previous=np.load(path)
        assert np.array_equal(previous['x'],x) and np.array_equal(previous['y'],y) and np.array_equal(previous['j'],j),'prepared resume mismatch'
    else:
        temporary=path.with_suffix('.tmp')
        with open(temporary,'wb') as stream:np.savez(stream,x=x,y=y,j=j)
        temporary.replace(path)
    return dict(path=str(path),family=m['family'],dof=m['dof'],episode=m['episode'],rows=len(x),sha256=p.digest(path))


def prepare(out):
    value=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'teacher-campaign/teacher_manifest.json').read_text())
    assert len(manifest)==1024 and len({m['family'] for m in manifest})==64
    directory=out/'prepared';directory.mkdir(exist_ok=True)
    paths=[];began=time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(prepare_one,[(out,directory,value['stride'],m) for m in manifest])):
            if result is not None:paths.append(result)
            if (i+1)%64==0:print(f'prepared {i+1}/1024 episodes',flush=True)
    p.atomic_json(directory/'manifest.json',paths)
    p.atomic_json(directory/'timing.json',dict(episodes=len(paths),wall_s=time.perf_counter()-began,workers=4,stride=value['stride']))


def train(out):
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    value=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'prepared/manifest.json').read_text())
    order=value['family_order']['train'];valid_families=set(order[::8])
    magnitude,rate=l.bound_values(r.BOUNDS)
    models=out/'models';models.mkdir(exist_ok=True)
    x,y,j,family,dof=[],[],[],[],[]
    for record in manifest:
        data=np.load(record['path']);count=len(data['y'])
        x.append(data['x']);y.append(data['y']);j.append(data['j'])
        family.append(np.full(count,record['family'],dtype=np.int64));dof.append(np.full(count,record['dof'],dtype=np.int8))
    x=np.concatenate(x);y=np.concatenate(y);j=np.concatenate(j);family=np.concatenate(family);dof=np.concatenate(dof)
    yn=y/np.asarray(magnitude,dtype=np.float32)[j];valid=np.isin(family,list(valid_families));fit=~valid
    allids=np.arange(len(x))
    ids3=allids[fit & (dof==3)];ids4=allids[fit & (dof==4)]
    rng=np.random.default_rng(20261022)
    matched={d:rng.choice(ids,size=min(value['matched_rows_per_embodiment'],len(ids)),replace=False) for d,ids in [(3,ids3),(4,ids4)]}
    budget=min(len(matched[3]),len(matched[4]));matched={d:ids[:budget] for d,ids in matched.items()}
    matched_union=np.concatenate([matched[3],matched[4]])
    p.atomic_json(models/'exposure.json',dict(rows_per_embodiment=budget,shared_union_sha256=hashlib.sha256(np.sort(matched_union).tobytes()).hexdigest(),
                  expert_union_sha256=hashlib.sha256(np.sort(np.concatenate(list(matched.values()))).tobytes()).hexdigest(),
                  shared_updates=2*value['matched_updates_per_embodiment'],expert_bank_updates=2*value['matched_updates_per_embodiment'],
                  interpretation='same union of training rows; equal total gradient updates and batch-size; expert bank has twice the parameter count',
                  full_data_rows=len(x),validation_families=sorted(valid_families)))
    tasks=[]
    for mode in ['matched','full']:
        for emb in [0,3,4]:tasks.append((f'{"shared" if emb==0 else "per_"+str(emb)}_{mode}',emb,64,mode))
    for count in [8,16,32]:tasks.append((f'curve_{count}',0,count,'full'))
    reports=[]
    for name,emb,count,mode in tasks:
        path=models/f'{name}.txt'
        report_path=models/f'{name}.json'
        if path.exists() and report_path.exists():
            previous=json.loads(report_path.read_text())
            assert previous['model_sha256']==p.digest(path) and previous['protocol_sha256']==p.digest(out/'protocol.json') and previous['prepared_manifest_sha256']==p.digest(out/'prepared/manifest.json'),'training resume identity mismatch'
            reports.append(previous);continue
        for partial in [path,report_path]:
            if partial.exists():partial.rename(partial.with_name(partial.name+f'.interrupted-{time.time_ns()}'))
        torch.manual_seed(20261022+emb)
        chosen=np.isin(family,order[:count]) & fit & ((dof==emb) if emb else True)
        train_ids=allids[chosen] if mode=='full' else matched_union if emb==0 else matched[emb]
        vi=allids[valid & ((dof==emb) if emb else True)]
        mean=x[train_ids].mean(0);scale=np.maximum(x[train_ids].std(0),.01)
        xx=torch.from_numpy((x-mean)/scale);yy=torch.from_numpy(yn[:,None])
        model=nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1),nn.Tanh())
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        generator=torch.Generator().manual_seed(20261022+emb)
        ti=torch.from_numpy(train_ids);v=torch.from_numpy(vi)
        history=[];started=time.perf_counter();best=math.inf;state=None;updates=0
        epochs=value['epochs'] if mode=='full' else 1
        for epoch in range(epochs):
            if mode=='matched':
                plans={d:torch.Generator().manual_seed(20261022+d) for d in [3,4]}
                batches=[]
                for step in range(value['matched_updates_per_embodiment']):
                    for d in ([3,4] if emb==0 else [emb]):
                        ids=torch.from_numpy(matched[d])
                        batches.append(ids[torch.randint(len(ids),(2048,),generator=plans[d])])
                exposure=hashlib.sha256();exposure_by_embodiment={}
                for d in ([3,4] if emb==0 else [emb]):
                    individual=hashlib.sha256()
                    for batch in batches[(0 if d==3 else 1)::2] if emb==0 else batches:
                        exposure.update(batch.numpy().tobytes());individual.update(batch.numpy().tobytes())
                    exposure_by_embodiment[str(d)]=individual.hexdigest()
            else:batches=ti[torch.randperm(len(ti),generator=generator)].split(2048)
            for batch in batches:
                optimizer.zero_grad();loss=((model(xx[batch])-yy[batch])**2).mean();loss.backward();optimizer.step();updates+=1
            with torch.no_grad():
                loss_sum=0.
                for batch in v.split(8192):loss_sum+=float(((model(xx[batch])-yy[batch])**2).sum())
                validation=loss_sum/len(v)
            history.append(validation)
            if validation<best:best=validation;state={k:t.detach().clone() for k,t in model.state_dict().items()}
        model.load_state_dict(state)
        chunks=[magnitude,rate,mean.tolist(),scale.tolist()]
        for layer in [model[0],model[2],model[4]]:chunks.extend([layer.weight.detach().numpy().ravel().tolist(),layer.bias.detach().numpy().ravel().tolist()])
        path.write_text(f'RELIC_MLP_V1 relic.local.v1 16 32 32 {emb}\n'+'\n'.join(' '.join(format(float(v),'.17g') for v in chunk) for chunk in chunks)+'\n')
        with torch.no_grad():
            validation_squared=[]
            for batch in v.split(8192):validation_squared.append((model(xx[batch]).numpy().ravel()-yn[batch.numpy()])**2)
        normalized_errors=np.concatenate(validation_squared)
        record=dict(prepared_manifest_sha256=p.digest(out/'prepared/manifest.json'),protocol_sha256=p.digest(out/'protocol.json'),bounds_sha256=p.digest(r.BOUNDS),feature_version='relic.local.v1',name=name,embodiment=emb,prefix_family_budget=count,fitting_family_count=len(set(family[train_ids].tolist())),validation_family_count=len(valid_families),total_exposed_family_count=len(set(family[train_ids].tolist())|valid_families),fit_families=sorted(set(family[train_ids].tolist())),
                    validation_families=sorted(valid_families),fit_rows=len(train_ids),validation_rows=len(vi),updates=updates,
                    validation_mse=best,validation_mse_rad2=float(np.mean(normalized_errors*np.asarray(magnitude)[j[vi]]**2)),zero_validation_mse_rad2=float(np.mean(y[vi]**2)),copy_previous_validation_mse_rad2=float(np.mean((x[vi,7]-y[vi])**2)),zero_validation_mse=float(np.mean(yn[vi]**2)),
                    copy_previous_validation_mse=float(np.mean((x[vi,7]/np.asarray(magnitude)[j[vi]]-yn[vi])**2)),
                    actual_exposure_sha256=exposure.hexdigest() if mode=='matched' else None,exposure_by_embodiment=exposure_by_embodiment if mode=='matched' else None,validation_curve=history,train_s=time.perf_counter()-started,model_sha256=p.digest(path))
        p.atomic_json(models/f'{name}.json',record);reports.append(record)
        print(f'{name}: {record["train_s"]:.1f}s MSE={best:.6f}, zero={record["zero_validation_mse"]:.6f}, copy={record["copy_previous_validation_mse"]:.6f}',flush=True)
    byname={record['name']:record for record in reports}
    for d in [3,4]:
        assert byname['shared_matched']['exposure_by_embodiment'][str(d)]==byname[f'per_{d}_matched']['exposure_by_embodiment'][str(d)], 'mismatched training exposure'
    p.atomic_json(models/'report.json',reports)


def evaluate(out,baselines_only=False):
    value=json.loads((out/'protocol.json').read_text());assert value==r.protocol()
    selection=[m for m in value['specs'] if m['partition'].endswith('_test')]
    assert len({m['family'] for m in selection})==40
    train_families=set(value['family_order']['train']);assert all(m['family'] not in train_families for m in selection)
    tuned=json.loads((out/'tuned.json').read_text());models=out/'models';tasks=[]
    for m in selection:
        methods=['conventional','analytical','gravity','integral','zero','copy_previous']
        if not baselines_only:methods+=['shared_matched','per_matched','shared_full','per_full','curve_8','curve_16','curve_32']
        for method in methods:
            model='none';control=None
            if method.startswith(('shared','curve')):model=models/f'{method}.txt'
            elif method.startswith('per_'):model=models/f'per_{m["dof"]}_{method.split("_")[-1]}.txt'
            if method in tuned:control=tuned[method]['control']
            tasks.append((out/'evaluation',m,method,model,control))
        if not baselines_only and m['dof']==4:tasks.append((out/'evaluation',m,'common_transfer',models/'per_3_full.txt',None))
    timing=out/('evaluation/baseline-timing.json' if baselines_only else 'evaluation/timing.json')
    previously_complete=timing.exists() and all((task[0]/task[2]/task[1]['episode']/'metadata.json').exists() for task in tasks)
    began=time.perf_counter();results=r.execute(tasks)
    p.atomic_json(out/('evaluation/baseline-index.json' if baselines_only else 'evaluation/index.json'),[dict(episode=m['episode'],controller=m['controller'],outcome=m['outcome']) for m in sorted(results,key=lambda m:(m['episode'],m['controller']))])
    target=timing.with_name('baseline-resume-timing.json' if baselines_only else 'resume-timing.json') if previously_complete else timing
    p.atomic_json(target,dict(episodes=len(results),cached_validation=previously_complete,wall_s=time.perf_counter()-began,protocol_sha256=p.digest(out/'protocol.json')))


def bootstrap(values,seed=20261022,reps=4000):
    import numpy as np
    vals=np.asarray(values,float);rng=np.random.default_rng(seed)
    estimates=vals[rng.integers(len(vals),size=(reps,len(vals)))].mean(1)
    return dict(mean=float(vals.mean()),ci95=[float(v) for v in np.quantile(estimates,[.025,.975])],families=len(vals))


def report(out):
    value=json.loads((out/'protocol.json').read_text());records={}
    for method in sorted((out/'evaluation').iterdir()):
        if method.is_dir():records[method.name]=sorted([json.loads(f.read_text()) for f in method.glob('r*/metadata.json')],key=lambda m:(m['family'],m['dof'],m['category']))
    expected_methods={'conventional','analytical','gravity','integral','zero','copy_previous','shared_matched','per_matched','shared_full','per_full','curve_8','curve_16','curve_32','common_transfer'}
    assert set(records)==expected_methods,'incomplete evaluation method matrix'
    specs=[m for m in value['specs'] if m['partition'].endswith('_test')]
    for method,rows in records.items():
        expected={m['episode'] for m in specs if method!='common_transfer' or m['dof']==4}
        assert len(rows)==len(expected) and {m['episode'] for m in rows}==expected,'missing or duplicated evaluation runs'
    conventional={m['episode']:m for m in records['conventional']}
    for method in ['zero','copy_previous']:
        assert all(m['trace_sha256']==conventional[m['episode']]['trace_sha256'] for m in records[method]),'zero/copy baseline diverged from conventional'
    restored_pairs=0
    for method,rows in records.items():
        by_condition={(m['family'],m['dof'],m['category']):m for m in rows}
        for m in rows:
            if m['category']=='restored_nominal':
                nominal=by_condition[(m['family'],m['dof'],'nominal')]
                assert m['physics']==nominal['physics'] and m['metrics']==nominal['metrics'],'nominal/restored control mismatch'
                restored_pairs+=1
    fingerprints=defaultdict(set)
    for method,rows in records.items():
        for m in rows:fingerprints[p.trajectory_key(m)].add(m['family'])
    assert all(len(fs)==1 for fs in fingerprints.values()),'trajectory fingerprint leakage: stop for review'
    train_fingerprints={p.trajectory_key(json.loads(f.read_text())) for f in (out/'teacher-campaign/teachers').glob('*/baseline/metadata.json')}
    assert not train_fingerprints & set(fingerprints),'training/evaluation trajectory leakage: stop for review'
    lookup={method:{m['episode']:m for m in rows} for method,rows in records.items()}
    summary={}
    for method,rows in records.items():
        primary=[m for m in rows if m['category'] in value['primary_categories']]
        entry=dict(episodes=len(rows),primary_episodes=len(primary),families=len({m['family'] for m in rows}),
                   outcomes=dict(Counter(m['outcome'] for m in rows)),fallback_episodes=sum(m['fallbacks']>0 for m in rows),
                   incomplete=sum(m['completed_ticks']!=m['planned_ticks'] for m in rows),inference_s=sum(m['inference_s'] for m in rows),
                   mean_inference_us_per_tick=1e6*sum(m['inference_s'] for m in rows)/max(1,sum(m['completed_ticks'] for m in rows)),
                   groups={})
        for group,selection in [('primary',primary)]+[(f'{key}:{val}',[m for m in rows if m[key]==val]) for key in ['dof','category','partition'] for val in sorted({m[key] for m in rows})]:
            byfamily=defaultdict(list);deltas=defaultdict(list);complete=[]
            for m in selection:
                base=conventional[m['episode']]
                for key in ['seed','family','dof','initial_q_rad','goal_q_rad','physics','command_ticks','planned_ticks','disturbance']:
                    assert m[key]==base[key],f'paired condition mismatch: {method} {key}'
                byfamily[m['family']].append(float(m['task_success']))
                if m['completed_ticks']==m['planned_ticks'] and base['completed_ticks']==base['planned_ticks']:
                    complete.append(m)
                    deltas[m['family']].append(m['metrics']['joint_rmse_rad']-base['metrics']['joint_rmse_rad'])
            detail=dict(episodes=len(selection),successes=sum(m['task_success'] for m in selection),
                        success=bootstrap([sum(v)/len(v) for v in byfamily.values()]),paired_incomplete_omitted=len(selection)-len(complete))
            if deltas:detail['paired_rmse_delta_vs_conventional']=bootstrap([sum(v)/len(v) for v in deltas.values()])
            for metric in ['joint_rmse_rad','peak_error_rad','command_end_cartesian_m','final_cartesian_m']:
                clusters=defaultdict(list)
                for m in complete:clusters[m['family']].append(m['metrics'][metric])
                if clusters:detail[metric]=bootstrap([sum(v)/len(v) for v in clusters.values()])
            detail['paired_comparisons']={}
            for reference in ['conventional','gravity','integral','shared_full','per_full']:
                rmse_delta=defaultdict(list);success_delta=defaultdict(list)
                for m in selection:
                    other=lookup[reference].get(m['episode'])
                    if other is None:continue
                    success_delta[m['family']].append(float(m['task_success'])-float(other['task_success']))
                    if m['completed_ticks']==m['planned_ticks'] and other['completed_ticks']==other['planned_ticks']:
                        rmse_delta[m['family']].append(m['metrics']['joint_rmse_rad']-other['metrics']['joint_rmse_rad'])
                pair=dict(complete_pairs=sum(len(v) for v in rmse_delta.values()))
                if rmse_delta:pair['rmse_delta_rad']=bootstrap([sum(v)/len(v) for v in rmse_delta.values()])
                if success_delta:pair['success_delta']=bootstrap([sum(v)/len(v) for v in success_delta.values()])
                detail['paired_comparisons'][reference]=pair
            if method=='common_transfer':
                clusters=defaultdict(list)
                for m in complete:clusters[m['family']].append(math.sqrt(sum(x*x for x in m['metrics']['joint_rmse_by_joint_rad'][:3])/3))
                if clusters:detail['common_joints_rmse_rad']=bootstrap([sum(v)/len(v) for v in clusters.values()])
                detail['common_joints_paired_comparisons']={}
                for reference in ['conventional','gravity','integral','shared_full','per_full']:
                    delta=defaultdict(list);base_common=defaultdict(list)
                    for m in complete:
                        other=lookup[reference][m['episode']]
                        ours=math.sqrt(sum(x*x for x in m['metrics']['joint_rmse_by_joint_rad'][:3])/3)
                        theirs=math.sqrt(sum(x*x for x in other['metrics']['joint_rmse_by_joint_rad'][:3])/3)
                        delta[m['family']].append(ours-theirs);base_common[m['family']].append(theirs)
                    if delta:detail['common_joints_paired_comparisons'][reference]=dict(delta_rad=bootstrap([sum(v)/len(v) for v in delta.values()]),reference_rad=bootstrap([sum(v)/len(v) for v in base_common.values()]))
            entry['groups'][group]=detail
        summary[method]=entry
    p.atomic_json(out/'comparison.json',dict(protocol_sha256=p.digest(out/'protocol.json'),methods=summary,unique_evaluation_fingerprints=len(fingerprints),train_test_fingerprint_overlap=0,nominal_restored_identical_pairs=restored_pairs,zero_copy_identical_to_conventional=True))
    for method,record in summary.items():
        row=record['groups']['primary'];print(method, row.get('joint_rmse_rad',{'undefined':'no complete pairs'}),row['successes'],'/',row['episodes'],flush=True)
