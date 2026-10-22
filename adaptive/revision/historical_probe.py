#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision as r
import pipeline as p


def main():
    selected={}
    for file in sorted((r.ROOT/'adaptive/runs/evaluation-final-executable').glob('e*/conventional/metadata.json')):
        m=json.loads(file.read_text());selected.setdefault((m['dof'],m['category']),m)
    assert len(selected)==16
    results=r.execute([(r.DEFAULT/'historical-replay',m,'conventional','none',None) for m in selected.values()])
    original={m['episode']:m for m in selected.values()}
    for m in results:
        assert m['trace_sha256']==original[m['episode']]['trace_sha256'],'conventional behavior changed'
        for key,value in original[m['episode']]['metrics'].items():assert m['metrics'][key]==value,key
    p.atomic_json(r.DEFAULT/'historical_replay.json',dict(episodes=len(results),byte_identical_traces=len(results),unchanged_metrics=True))
    print(f'{len(results)} historical conventional traces reproduce byte-for-byte')

if __name__=='__main__':main()
