#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import learning as l
import pipeline as p
import revision as r
import revision_learning as rl


def main():
    import numpy as np
    directory=r.DEFAULT/'training-qualification';directory.mkdir(exist_ok=False)
    prepared=directory/'prepared';prepared.mkdir()
    protocol=json.loads((r.DEFAULT/'protocol.json').read_text())
    protocol.update(epochs=2,matched_updates_per_embodiment=8,matched_rows_per_embodiment=2000)
    p.atomic_json(directory/'protocol.json',protocol)
    teachers=json.loads((r.DEFAULT/'teacher-qualification/teacher_manifest.json').read_text())
    manifest=[]
    for m in teachers:
        best=Path(m['accepted'][-1]);targets,_=l.table_from_trace(best)
        rows=[]
        for source in [best.parent/'baseline',best]:rows.extend(l.examples_from_trace(source,stride=64,targets=targets))
        path=prepared/f'{m["episode"]}.npz'
        np.savez(path,x=np.asarray([x for x,_,_ in rows],np.float32),y=np.asarray([y for _,y,_ in rows],np.float32),j=np.asarray([j for _,_,j in rows],np.int8))
        manifest.append(dict(path=str(path),family=m['family'],dof=m['dof']))
    p.atomic_json(prepared/'manifest.json',manifest)
    rl.train(directory)
    results=[]
    for m in r.protocol()['specs'][:16]:
        native=l.run_native(r.DEFAULT/'generator-pinned',r.ROOT/'examples',directory/'execution'/m['episode'],m,'mlp',r.BOUNDS,directory/'models/shared_full.txt')
        assert native['fallbacks']==0 and native['completed_ticks']==native['planned_ticks']
        results.append(dict(episode=m['episode'],outcome=native['outcome'],joint_rmse_rad=native['metrics']['joint_rmse_rad']))
    p.atomic_json(directory/'qualification.json',dict(episodes=len(results),results=results,scope='integration-only two-epoch probe; no accuracy acceptance criterion; main protocol unchanged'))

if __name__=='__main__':main()
