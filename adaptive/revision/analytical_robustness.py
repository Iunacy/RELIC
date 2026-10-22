#!/usr/bin/env python3
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision as r
import revision_learning as rl
import pipeline as p


def main():
    out=r.DEFAULT;value=r.protocol()
    grid=[dict(lead=lead,ki=ki,kp=kp,integral_limit=.2) for lead in [.1,.15] for ki in [1.,2.,4.] for kp in [.2,.5,1.]]
    plan=dict(grid=grid,selection=value['tuning'],development_families=value['family_order']['development'],
              rationale='integral state cap .2 rad permits use of every per-joint applied-correction bound; applied limits remain identical',
              status='exploratory robustness analysis registered after the single predeclared severe demo, before full-cohort scores examined; primary protocol retained',
              unchanged=['models','success thresholds','ILC gate','evaluation families','safety monitoring'])
    registration=out/'analytical_robustness_protocol.json'
    if registration.exists():assert json.loads(registration.read_text())==plan
    else:p.atomic_json(registration,plan)
    development=[m for m in value['specs'] if m['partition']=='development']
    records=[];began=time.perf_counter()
    for i,config in enumerate(grid):
        control=out/f'robustness-{i}.control'
        text=f"RELIC_CONTROL_V1 {config['lead']} {config['ki']} {config['kp']} .2 -1\n"
        if control.exists():assert control.read_text()==text
        else:control.write_text(text)
        rows=r.execute([(out/f'robustness-tuning-{i}',m,'integral','none',control) for m in development])
        safe=all(m['completed_ticks']==m['planned_ticks'] and m['outcome'] in ['Success','NonConvergence'] and m['fallbacks']==0 for m in rows)
        records.append(dict(index=i,config=config,control=str(control),safe=safe,mean_rmse=sum(m['metrics']['joint_rmse_rad'] for m in rows)/len(rows)))
        p.atomic_json(out/'analytical_robustness_tuning.json',records)
    eligible=[m for m in records if m['safe']]
    if not eligible:
        p.atomic_json(out/'analytical_robustness.json',dict(status='no eligible development configuration',candidates=records))
        return
    chosen=min(eligible,key=lambda m:m['mean_rmse'])
    p.atomic_json(out/'analytical_robustness_selected.json',chosen)
    evaluation=[m for m in value['specs'] if m['partition'].endswith('_test')]
    rows=r.execute([(out/'robustness-evaluation',m,'integral','none',chosen['control']) for m in evaluation])
    while not (out/'comparison.json').exists():time.sleep(5)
    primary=[m for m in rows if m['category'] in value['primary_categories']]
    result=dict(status='exploratory robustness',selected=chosen,episodes=len(rows),primary_episodes=len(primary),
                families=len({m['family'] for m in rows}),outcomes=dict(Counter(m['outcome'] for m in rows)),groups={},wall_s=time.perf_counter()-began)
    for name,selection in [('primary',primary)]+[(f'{key}:{v}',[m for m in rows if m[key]==v]) for key in ['dof','category','partition'] for v in sorted({m[key] for m in rows})]:
        entry=dict(episodes=len(selection),successes=sum(m['task_success'] for m in selection),paired={})
        success=defaultdict(list)
        for m in selection:success[m['family']].append(float(m['task_success']))
        entry['success']=rl.bootstrap([sum(v)/len(v) for _,v in sorted(success.items())])
        for metric in ['joint_rmse_rad','peak_error_rad','command_end_cartesian_m','final_cartesian_m']:
            clusters=defaultdict(list)
            for m in selection:
                if m['completed_ticks']==m['planned_ticks']:clusters[m['family']].append(m['metrics'][metric])
            if clusters:entry[metric]=rl.bootstrap([sum(v)/len(v) for _,v in sorted(clusters.items())])
        for method in ['conventional','gravity','integral','shared_full','per_full','shared_matched','per_matched']:
            delta=defaultdict(list);success_delta=defaultdict(list)
            for m in selection:
                reference=json.loads((out/'evaluation'/method/m['episode']/'metadata.json').read_text())
                for key in ['seed','family','dof','physics','initial_q_rad','goal_q_rad','planned_ticks']:assert m[key]==reference[key]
                success_delta[m['family']].append(float(m['task_success'])-float(reference['task_success']))
                if m['completed_ticks']==m['planned_ticks'] and reference['completed_ticks']==reference['planned_ticks']:
                    delta[m['family']].append(m['metrics']['joint_rmse_rad']-reference['metrics']['joint_rmse_rad'])
            item=dict(complete_pairs=sum(map(len,delta.values())),success_delta=rl.bootstrap([sum(v)/len(v) for _,v in sorted(success_delta.items())]))
            if delta:item['rmse_delta_rad']=rl.bootstrap([sum(v)/len(v) for _,v in sorted(delta.items())])
            entry['paired'][method]=item
        result['groups'][name]=entry
    p.atomic_json(out/'analytical_robustness.json',result)
    print(json.dumps(result['groups']['primary'],indent=2),flush=True)

if __name__=='__main__':main()
