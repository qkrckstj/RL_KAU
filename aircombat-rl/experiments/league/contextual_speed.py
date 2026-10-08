"""Optional public-state speed adjustment after the fixed probe.

This experimental family is not part of the running temporal-repair search.
It preserves heading commands and only increases the tracking speed target
when elapsed time, distance and observed enemy speed meet learned thresholds.
Own firing solutions and the existing defensive maneuver retain base control.
"""
import json
import math
import zipfile
import numpy as np
from experiments.league.separated_probe import Policy as Separated
from experiments.league.controller import Policy as Expert

CONTEXT_NAMES=('activation_seconds','minimum_range','enemy_speed_threshold','speed_gain')
CONTEXT_LOW=np.array([.5,0.,300.,0.])
CONTEXT_HIGH=np.array([45.,4000.,650.,2.])


class ContextExpert:
    def __init__(self, parent, context):
        self.base=Expert(parameters=parent)
        self.context=np.asarray(context,float).copy()
        if (self.context.shape!=CONTEXT_LOW.shape or not np.isfinite(self.context).all()
                or np.any(self.context<CONTEXT_LOW) or np.any(self.context>CONTEXT_HIGH)):
            raise ValueError('Expected four bounded finite context coefficients')
        if self.base.parameters[0]!=0:
            raise ValueError('Contextual post-probe parent must have no scripted opening')

    def act(self,obs):
        action=self.base.act(obs)
        delay,min_range,threshold,gain=self.context
        if gain==0 or 120.-float(obs[38])<delay or float(obs[34])>=.5:
            return action
        dx,dy=float(obs[15]-obs[0]),float(obs[16]-obs[1])
        distance=math.hypot(dx,dy)
        enemy_speed=math.sqrt(sum(float(v)**2 for v in obs[18:21]))/.514444
        if distance<min_range or enemy_speed<threshold:return action
        parent=self.base.parameters
        angle=math.atan2(-dx,-dy)-float(obs[26])
        enemy_error=abs(math.degrees(math.atan2(math.sin(angle),math.cos(angle))))
        if distance<parent[11] and enemy_error<parent[12]:return action
        base_speed=parent[10] if parent[9]>0 and distance>parent[9] else parent[2]
        target=min(650.,max(base_speed,parent[2]+gain*max(0.,enemy_speed-parent[2])))
        own_speed=math.sqrt(sum(float(v)**2 for v in obs[3:6]))/.514444
        throttle=2 if own_speed<target-5 else 0 if own_speed>target+5 else 1
        return int(3*(action//3)+throttle)


class ContextParentAfterProbe:
    def __init__(self,owner,configuration):
        self.owner=owner
        self.probe=Expert(parameters=configuration['probe'])
        self.parent=ContextExpert(configuration['parent'],configuration['context'])

    def act(self,obs):
        return (self.probe if self.owner.selected is None else self.parent).act(obs)


class Policy(Separated):
    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as archive:
                configuration=json.loads(archive.read('parameters.json'))
        if set(configuration)!={'probe','parent','interceptor','context'}:
            raise ValueError('Unexpected contextual policy configuration')
        super().__init__(configuration={k:configuration[k] for k in ('probe','parent','interceptor')})
        self.experts[0]=ContextParentAfterProbe(self,configuration)

    def __str__(self):
        return 'Separated probe with public-state-dependent tracking speed'
