#!/usr/bin/env python3
import concurrent.futures as futures
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision as r
import pipeline as p


def main():
    out=r.DEFAULT;value=r.protocol();selected=[]
    for i,family in enumerate(value['family_order']['train'][:32]):
        category=value['categories'][i%8]
        selected.extend(m for m in value['specs'] if m['family']==family and m['category']==category)
    assert len(selected)==64 and len({m['family'] for m in selected})==32
    registration=dict(episodes=[m['episode'] for m in selected],base_families=32,dofs=[3,4],iterations=8,lead_ticks=[0,20],
                      selection='first 32 hash-ranked training families, category cyclically balanced; both embodiments',
                      use='convergence replication only; never added to model fitting; original two-family probe retained',
                      acceptance=value['acceptance'])
    path=out/'convergence_replication_protocol.json'
    if path.exists():assert json.loads(path.read_text())==registration
    else:p.atomic_json(path,registration)
    for lead in [0,20]:
        directory=out/f'convergence-expanded-lead-{lead}';directory.mkdir(exist_ok=True)
        began=time.perf_counter()
        with futures.ProcessPoolExecutor(max_workers=6) as pool:
            results=[]
            for result in pool.map(r.teacher_task,[(directory,m,8,lead) for m in selected]):
                results.append(result)
                p.atomic_json(directory/'teacher_manifest.json',results)
                if len(results)%8==0:print(f'lead {lead}: {len(results)}/64 episodes',flush=True)
        r.summarize_teachers(directory,results,time.perf_counter()-began)
        rows=[]
        for i in range(8):
            rows.append(dict(iteration=i+1,mean_proposal_rmse=sum(m['attempts'][i]['trial_rmse'] for m in results)/len(results),
                             mean_best_rmse=sum(m['attempts'][i]['best_rmse'] for m in results)/len(results),
                             accepted=sum(m['attempts'][i]['accepted'] for m in results),rejected=sum(not m['attempts'][i]['accepted'] for m in results),
                             magnitude_saturated_samples=sum(m['attempts'][i]['magnitude_saturated_samples'] for m in results),
                             rate_saturated_samples=sum(m['attempts'][i]['rate_saturated_samples'] for m in results),
                             total_joint_ticks=sum(m['attempts'][i]['total_joint_ticks'] for m in results)))
        p.atomic_json(directory/'convergence.json',rows)

if __name__=='__main__':main()
