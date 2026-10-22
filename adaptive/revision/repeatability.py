#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision as r
import pipeline as p


def main():
    out=r.DEFAULT;value=r.protocol()
    families={value['family_order'][name][0] for name in ['trajectory_test','combination_test','severity_test','transfer_test']}
    selection=[m for m in value['specs'] if m['family'] in families and m['category'] in ['nominal','combined']]
    assert len(selection)==16
    tuned=json.loads((out/'tuned.json').read_text());tasks=[]
    for m in selection:
        for method in ['conventional','gravity','integral','shared_full','per_full']:
            model=out/'models/shared_full.txt' if method=='shared_full' else out/f'models/per_{m["dof"]}_full.txt' if method=='per_full' else 'none'
            control=tuned[method]['control'] if method in tuned else None
            tasks.append((out/'repeatability',m,method,model,control))
    r.execute(tasks)
    hashes=[]
    for _,m,method,_,_ in tasks:
        original=json.loads((out/'evaluation'/method/m['episode']/'metadata.json').read_text())
        repeat=json.loads((out/'repeatability'/method/m['episode']/'metadata.json').read_text())
        assert original['trace_sha256']==repeat['trace_sha256'] and original['metrics']==repeat['metrics'],'repeatability failure'
        hashes.append(dict(episode=m['episode'],method=method,trace_sha256=repeat['trace_sha256']))
    p.atomic_json(out/'repeatability.json',dict(episodes=len(hashes),conditions=16,base_families=4,byte_identical=len(hashes),hashes=hashes))
    print(f'{len(hashes)} controller runs reproduce byte-for-byte')

if __name__=='__main__':main()
