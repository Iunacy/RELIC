import argparse
import json
import time
import uuid
from pathlib import Path

from .arm import Arm, ROOT
from .engine import comparison


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',choices=['menagerie','synthetic','both'],default='both')
    p.add_argument('--output',type=Path,default=ROOT/'runs'/('qualification-'+uuid.uuid4().hex))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    for robot in (['menagerie','synthetic'] if args.robot=='both' else [args.robot]):
        summary=[]
        for degraded in (False,True):
            for target in ('target_left','target_center','target_right'):
                out=args.output/f'{robot}-{target}-{degraded}'
                report=comparison(robot,target,degraded,out,capture=False)
                for control in report['controllers']:
                    summary.append(control)
                    print(robot,target,degraded,control['controller'],control['status'],control['metrics'],flush=True)
        path=args.output/(robot+'_qualification.json')
        path.write_text(json.dumps(summary,indent=2)+'\n')
        print('summary',path,flush=True)
    (args.output/'timing.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-start)))


if __name__=='__main__':
    main()
