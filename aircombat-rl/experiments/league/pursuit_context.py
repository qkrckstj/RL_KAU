"""Gate the contextual speed boost by the opponent's public motion direction."""
import json
import math
import zipfile
import numpy as np
from experiments.league.separated_probe import Policy as Separated
from experiments.league.controller import Policy as Expert
from experiments.league.contextual_speed import ContextExpert

PURSUIT_NAMES=('activation_seconds','minimum_range','enemy_speed_threshold','speed_gain','minimum_away_cosine')
PURSUIT_LOW=np.array([.5,0.,300.,0.,-1.])
PURSUIT_HIGH=np.array([45.,4000.,650.,2.,1.])


class PursuitExpert:
    def __init__(self,parent,context):
        self.context=np.asarray(context,float).copy()
        if (self.context.shape!=PURSUIT_LOW.shape or not np.isfinite(self.context).all()
                or np.any(self.context<PURSUIT_LOW) or np.any(self.context>PURSUIT_HIGH)):
            raise ValueError('Expected five bounded finite context coefficients')
        self.speed=ContextExpert(parent,self.context[:4])

    def act(self,obs):
        threshold=self.context[4]
        if threshold<=-1.:
            return self.speed.act(obs)
        dx,dy=float(obs[15]-obs[0]),float(obs[16]-obs[1])
        vx,vy=float(obs[18]),float(obs[19])
        denominator=math.hypot(dx,dy)*math.hypot(vx,vy)
        if denominator<=1e-12:
            return self.speed.base.act(obs)
        away=max(-1.,min(1.,(dx*vx+dy*vy)/denominator))
        if away<threshold:
            return self.speed.base.act(obs)
        return self.speed.act(obs)


class PursuitParentAfterProbe:
    def __init__(self,owner,configuration):
        self.owner=owner
        self.probe=Expert(parameters=configuration['probe'])
        self.parent=PursuitExpert(configuration['parent'],configuration['context'])

    def act(self,obs):
        return (self.probe if self.owner.selected is None else self.parent).act(obs)


class Policy(Separated):
    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as archive:
                configuration=json.loads(archive.read('parameters.json'))
        if set(configuration)!={'probe','parent','interceptor','context'}:
            raise ValueError('Unexpected pursuit-context configuration')
        super().__init__(configuration={k:configuration[k] for k in ('probe','parent','interceptor')})
        self.experts[0]=PursuitParentAfterProbe(self,configuration)

    def __str__(self):
        return 'Separated probe with motion-gated contextual tracking speed'
