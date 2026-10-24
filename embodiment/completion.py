"""Independent physical terminal-state admission for versioned new collections."""
import math
from .benchmark import COLORS, TASK_GRIPPER_LIMITS, inside_bin

CONTRACT = 'viper.physical_episode.v2'


def validate_completion(metadata, outcome, rows, telemetry):
    if metadata.get('episode_contract') != CONTRACT: return {'version': 'legacy', 'terminal_dwell_verified': False}
    if len(rows) != len(telemetry) or len(rows) < 2: raise ValueError('new physical episode lacks aligned observations')
    robot, target = metadata['embodiment'], metadata['target']
    from .contract import manifest
    import json
    if metadata.get('robot_contract') != json.loads(json.dumps(manifest(robot))): raise ValueError('recorded robot contract differs from frozen mapping')
    for row, physical in zip(rows, telemetry, strict=True):
        if abs(row['timestamp']-physical['timestamp']) > 1e-8: raise ValueError('terminal telemetry timestamp mismatch')
        if set(physical.get('cube_speeds', {})) != set(COLORS): raise ValueError('measured object speeds missing')
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in physical['cube_speeds'].values()):
            raise ValueError('invalid object speeds')
    if not outcome['success']: return {'version': CONTRACT, 'terminal_dwell_verified': False}
    closed, opened = TASK_GRIPPER_LIMITS[robot]
    end = rows[-1]['timestamp']
    tail = [(row, physical) for row, physical in zip(rows, telemetry, strict=True) if row['timestamp'] >= end-1.-1e-8]
    if not tail or end-tail[0][0]['timestamp'] < 1.-1e-8: raise ValueError('one-second terminal stability evidence missing')
    for row, physical in tail:
        aperture = (row['joint_position'][-2]-closed)/(opened-closed)
        if (not inside_bin(physical['cube_poses'][target]) or physical['cube_speeds'][target] >= .02 or
                physical['contacts'][target] or aperture < .84 or physical['unsafe'] or physical['safety_events']):
            raise ValueError('placement is not stably released for one second')
    if rows[-1]['ee_position'][2] < .08: raise ValueError('terminal tool did not retreat')
    return {'version': CONTRACT, 'terminal_dwell_verified': True}
