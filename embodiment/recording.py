"""Synchronized simulation observations/actions; images are real camera frames only."""
import json
from pathlib import Path
from .benchmark import Action, COLORS, finite


def validate_sample(row, previous_time=-1):
    required = {'timestamp','frame','joint_position','joint_velocity','ee_position',
                'action','instruction','phase','success'}
    if set(row) != required:
        raise ValueError('sample fields differ from schema')
    finite((row['timestamp'],),1)
    if row['timestamp'] <= previous_time:
        raise ValueError('timestamps must increase')
    if not isinstance(row['instruction'],str) or not row['instruction']:
        raise ValueError('missing instruction')
    if type(row['success']) is not bool or not isinstance(row['phase'],str):
        raise ValueError('invalid labels')
    finite(row['ee_position'],3)
    if len(row['joint_position']) < 5:
        raise ValueError('missing robot state')
    finite(row['joint_position'],len(row['joint_position']))
    finite(row['joint_velocity'],len(row['joint_position']))
    if not isinstance(row['action'],list) or len(row['action']) != 4:
        raise ValueError('expected delta_x,delta_y,delta_z,gripper')
    Action(tuple(row['action'][:3]),row['action'][3])
    frame = row['frame']
    if frame is not None and (not isinstance(frame,str) or Path(frame).is_absolute() or '..' in Path(frame).parts):
        raise ValueError('invalid camera frame path')
    return row['timestamp']


def validate_telemetry(row):
    finite((row['timestamp'],), 1)
    if set(row['cube_poses']) != set(COLORS) or type(row['unsafe']) is not bool:
        raise ValueError('invalid physical telemetry')
    if not isinstance(row['safety_events'], list) or any(not isinstance(event, str) for event in row['safety_events']):
        raise ValueError('invalid telemetry safety events')
    for pose in row['cube_poses'].values():
        finite(pose, 7)
    if set(row['contacts']) != set(COLORS) or any(type(value) is not bool for value in row['contacts'].values()):
        raise ValueError('invalid contact labels')
    if set(row['contact_forces']) != set(COLORS):
        raise ValueError('missing bilateral contact forces')
    for forces in row['contact_forces'].values():
        finite(forces, 2)
        if any(force < 0 for force in forces):
            raise ValueError('negative contact force magnitude')
    return row


class Recorder:
    def __init__(self,path,instruction,metadata=None):
        self.path = Path(path)
        self.path.mkdir(parents=True,exist_ok=False)
        self.instruction = instruction
        self.last_time = -1
        self.count = 0
        self.file = (self.path/'samples.jsonl').open('x')
        self.telemetry = (self.path/'telemetry.jsonl').open('x')
        (self.path/'metadata.json').write_text(json.dumps({
            'schema_version':1,'time_unit':'second','position_unit':'meter',
            'action_fields':['delta_x','delta_y','delta_z','gripper'],
            'action_frame':'world','action_alignment':'observation_to_next_control_tick',
            'gripper_range':[0,1],'gripper_open':1,**(metadata or {})},indent=2)+'\n')

    def append(self,obs,action,phase,success=False,frame=None):
        # action is the command applied from this observation to the next control tick.
        filename = None
        if frame is not None:
            filename = f'camera/{self.count:06d}.png'
            dest = self.path/filename
            dest.parent.mkdir(exist_ok=True)
            from PIL import Image
            Image.fromarray(frame).save(dest)
        row = dict(timestamp=obs.time,frame=filename,joint_position=list(obs.joints),
                   joint_velocity=list(obs.velocities),ee_position=list(obs.ee),
                   action=[*action.delta,action.gripper],instruction=self.instruction,
                   phase=phase,success=success)
        self.last_time = validate_sample(row,self.last_time)
        self.file.write(json.dumps(row,allow_nan=False)+'\n')
        self.file.flush()
        self.telemetry.write(json.dumps({'timestamp':obs.time,'cube_poses':obs.cubes,'contacts':obs.contacts,'contact_forces':obs.contact_forces,
                                         'unsafe':obs.unsafe,'safety_events':list(obs.safety_events)},allow_nan=False)+'\n')
        self.telemetry.flush()
        self.count += 1

    def close(self,result):
        self.file.close()
        self.telemetry.close()
        temporary = self.path/'outcome.json.tmp'
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        temporary.replace(self.path/'outcome.json')


    def abort(self):
        self.file.close()
        self.telemetry.close()


def lerobot_rows(path):
    """Validated export boundary for a future pinned LeRobot dataset writer.

    No dataset is fabricated. Missing-camera episodes are rejected for VLA export.
    Episode-wide outcome labels come from the final physical result. Map returned
    rows into LeRobotDataset.add_frame and save_episode once its version is selected.
    """
    path = Path(path)
    outcome = json.loads((path/'outcome.json').read_text())
    last = -1
    for line in (path/'samples.jsonl').read_text().splitlines():
        row = json.loads(line)
        last = validate_sample(row,last)
        if row['frame'] is None or not (path/row['frame']).is_file():
            raise ValueError('VLA export requires synchronized camera data')
        yield {'observation.state':row['joint_position']+row['joint_velocity'],
               'observation.images.overhead':str(path/row['frame']),
               'action':row['action'],'task':row['instruction'],
               'timestamp':row['timestamp'],'next.success':outcome['success']}
