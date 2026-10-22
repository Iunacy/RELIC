#!/usr/bin/env python3
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision as r
import revision_learning as rl
import pipeline as p


def summarize(out,prefix):
    records={lead:json.loads((out/f'{prefix}{lead}'/'teacher_manifest.json').read_text()) for lead in [0,20]}
    indexed={lead:{m['episode']:m for m in rows} for lead,rows in records.items()}
    assert indexed[0].keys()==indexed[20].keys()
    families={m['family'] for m in records[0]};curves=[]
    for i in range(8):
        groups={lead:defaultdict(list) for lead in [0,20]};paired=defaultdict(list)
        outcomes={lead:Counter() for lead in [0,20]};incomplete={0:0,20:0}
        for episode,a in indexed[0].items():
            b=indexed[20][episode]
            assert all(a[k]==b[k] for k in ['family','dof','category','severity'])
            trials={lead:json.loads((Path(indexed[lead][episode]['attempts'][i]['path'])/'metadata.json').read_text()) for lead in [0,20]}
            for lead,trial in trials.items():
                outcomes[lead][trial['outcome']]+=1
                if trial['completed_ticks']==trial['planned_ticks']:groups[lead][a['family']].append(trial['metrics']['joint_rmse_rad'])
                else:incomplete[lead]+=1
            if all(m['completed_ticks']==m['planned_ticks'] for m in trials.values()):
                paired[a['family']].append(trials[20]['metrics']['joint_rmse_rad']-trials[0]['metrics']['joint_rmse_rad'])
        row=dict(iteration=i+1,outcomes=outcomes,incomplete=incomplete,lead={})
        for lead in [0,20]:
            attempts=[m['attempts'][i] for m in records[lead]]
            accepted=sum(a['accepted'] for a in attempts)
            total=sum(a.get('total_joint_ticks',0) for a in attempts)
            row['lead'][lead]=dict(proposed_rmse_rad=rl.bootstrap([sum(v)/len(v) for v in groups[lead].values()]),
                accepted=accepted,rejected=len(attempts)-accepted,rejection_reasons=dict(Counter(a['reason'] for a in attempts if not a['accepted'])),
                proposal_magnitude_saturation_rate=sum(a.get('magnitude_saturated_samples',0) for a in attempts)/max(1,total),
                proposal_rate_saturation_rate=sum(a.get('rate_saturated_samples',0) for a in attempts)/max(1,total))
        if paired:row['paired_lead_minus_zero_rmse_rad']=rl.bootstrap([sum(v)/len(v) for v in paired.values()])
        curves.append(row)
    return dict(base_families=len(families),conditions=len(records[0]),curves=curves,
                interpretation='fixed-iteration offline training-only probe; accepted-best monotonicity follows the acceptance gate, not a convergence proof')


def main():
    value={name:summarize(r.DEFAULT,prefix) for name,prefix in [('original','convergence-lead-'),('expanded','convergence-expanded-lead-')]}
    p.atomic_json(r.DEFAULT/'convergence_comparison.json',value)
    print(json.dumps({name:dict(base_families=v['base_families'],final=v['curves'][-1]) for name,v in value.items()},indent=2))

if __name__=='__main__':main()
