#!/usr/bin/env python3
import math
import pipeline as p


def geometry(family,dof):
    state=family
    def uniform(lo,hi):
        nonlocal state
        mask=(1<<64)-1
        state=(state+0x9e3779b97f4a7c15)&mask
        z=state;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask;z=((z^(z>>27))*0x94d049bb133111eb)&mask
        value=(z^(z>>31))>>11
        return lo+(hi-lo)*value*2**-53
    initial=[uniform(-.9,.9),uniform(-.85,-.3),uniform(.05,.65)]
    goal=[uniform(-.9,.9),uniform(-.85,-.3),uniform(.05,.65)]
    if dof==4:initial.append(uniform(-.7,.7));goal.append(uniform(-.7,.7))
    duration=uniform(.8,1.8);vmax=uniform(.45,1.1);amax=uniform(1.2,3)
    for a,b in zip(initial,goal):duration=max(duration,1.875*abs(b-a)/vmax,math.sqrt(5.774*abs(b-a)/amax))
    return dict(family=family,dof=dof,initial_q_rad=initial,goal_q_rad=goal,timestep_s=.002,command_ticks=math.ceil(duration/.002),velocity_limit_rad_s=vmax,acceleration_limit_rad_s2=amax)


def audit(value):
    seen={};seeds={}
    for spec in value['specs']:
        if spec['seed'] in seeds:assert seeds[spec['seed']]==spec['partition'],'seed crosses partitions: stop for review'
        else:seeds[spec['seed']]=spec['partition']
        key=(spec['family'],spec['dof'])
        if key in seen:assert seen[key]['partition']==spec['partition'],'family crosses partitions: stop for review'
        else:seen[key]=dict(geometry(*key),partition=spec['partition'])
    rows=list(seen.values());nearest=math.inf;compared=0
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['dof']!=b['dof'] or a['partition']==b['partition']:continue
            delta=max(abs(x-y) for x,y in zip(a['initial_q_rad']+a['goal_q_rad'],b['initial_q_rad']+b['goal_q_rad']))
            nearest=min(nearest,delta);compared+=1
            assert delta>=.05 or abs(a['command_ticks']-b['command_ticks'])*.002>=.1,'near-duplicate cross-partition trajectory: stop for review'
            assert p.trajectory_key(a)!=p.trajectory_key(b),'cross-partition fingerprint leakage: stop for review'
    coverage={}
    for partition in sorted({m['partition'] for m in rows}):
        subset=[m for m in rows if m['partition']==partition]
        coverage[partition]=dict(base_families=len({m['family'] for m in subset}),embodied_families=len(subset),
            duration_range_s=[min(m['command_ticks']*.002 for m in subset),max(m['command_ticks']*.002 for m in subset)],
            positive_motion_by_joint=[sum(m['goal_q_rad'][j]>m['initial_q_rad'][j] for m in subset if m['dof']>j) for j in range(4)],
            negative_motion_by_joint=[sum(m['goal_q_rad'][j]<m['initial_q_rad'][j] for m in subset if m['dof']>j) for j in range(4)])
    return dict(coverage=coverage,embodied_families=len(rows),cross_partition_pairs=compared,minimum_endpoint_max_norm_rad=nearest,
                near_duplicate_rule='max endpoint joint distance < .05 rad AND duration difference < .1 s',
                fingerprints=[dict(family=m['family'],dof=m['dof'],partition=m['partition'],fingerprint=p.trajectory_key(m)) for m in rows])
