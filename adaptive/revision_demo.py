#!/usr/bin/env python3
import json
import argparse
import shutil
from pathlib import Path
import revision as r
import pipeline as p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=(args.output or (r.ROOT/'adaptive/runs/revision-demo-20261022' if args.fixtures else r.DEFAULT)).resolve()
    if args.fixtures:
        r.freeze(out)
        models=out/'models';models.mkdir(exist_ok=True)
        source=r.ROOT/'adaptive/revision/models/shared_full.txt';target=models/'shared_full.txt'
        if target.exists():assert p.digest(target)==p.digest(source),'existing demo model differs'
        else:shutil.copy2(source,target)
        tuning=json.loads((r.ROOT/'adaptive/revision/analytical.json').read_text())
        tuned={}
        for method,config in tuning.items():
            control=r.config_file(out,method,config['lead'],config['ki'],config['kp'])
            tuned[method]=dict(control=str(control))
        p.atomic_json(out/'tuned.json',tuned)
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
