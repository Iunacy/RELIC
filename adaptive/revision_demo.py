#!/usr/bin/env python3
import json
from pathlib import Path
import revision as r
import pipeline as p


def main():
    out=r.DEFAULT
    value=json.loads((out/'protocol.json').read_text())
    family=value['family_order']['severity_test'][0]
    spec=next(m for m in value['specs'] if m['family']==family and m['dof']==4 and m['category']=='combined')
    tuned=json.loads((out/'tuned.json').read_text())
    result=[]
    for method in ['conventional','gravity','integral','shared_full']:
        model=out/'models/shared_full.txt' if method=='shared_full' else 'none'
        control=tuned[method]['control'] if method in tuned else None
        meta=r.run_task((out/'demo',spec,method,model,control))
        result.append(dict(method=method,outcome=meta['outcome'],metrics=meta['metrics'],
                           fallbacks=meta['fallbacks'],termination_reason=meta['termination_reason'],
                           telemetry=str(out/'demo'/method/spec['episode']/'telemetry.csv.gz')))
    feedback=dict(selection='predeclared first severity-test family, 4dof combined fault',spec=spec,controllers=result)
    p.atomic_json(out/'demo_feedback.json',feedback)
    print(json.dumps(feedback,indent=2))

if __name__=='__main__':main()
