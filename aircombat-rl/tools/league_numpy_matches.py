"""Candidate backend for preverified NumPy-only submissions.

This is not used by frozen active experiments. A future scheduler must route
neural or unknown submissions to normal workers. Pure-policy code signatures
are explicit inputs; dependency detection is not inferred from a policy name.
"""
import hashlib
import json
import sys
import time
from tools.autolab_cem import ROOT, sha
from tools.league_matches import Actor, MatchEnv as OriginalMatchEnv, summary
from aircombat_gym.wvr.envs.fair import FairFightEnv
from tools.grade import play
from tools.policies import load


def code_signature(spec):
    if spec['kind'] != 'submission':
        raise ValueError('Only submitted policies have a source signature')
    folder=ROOT/spec['design']
    sources={path.name:sha(path) for path in sorted(folder.glob('*.py'))}
    if 'policy.py' not in sources:
        raise ValueError('Missing submitted policy source')
    return hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()


def numpy_capable(spec,approved_signatures):
    if spec['kind'] in ('bot','fixed','cem','reactive'):
        return True
    return spec['kind']=='submission' and code_signature(spec) in approved_signatures


class NumpyActor(Actor):
    def __init__(self,spec,approved_signatures):
        if not numpy_capable(spec,approved_signatures):
            raise ValueError('Neural or unknown submission needs a normal worker')
        if spec['kind']!='submission':
            super().__init__(spec)
            return
        if 'torch' in sys.modules:
            raise RuntimeError('NumPy-only worker already imported torch')
        self.spec=spec; self.mode='discrete'; self.bot=None
        self.predict,_,self.mode=load(ROOT/spec['design'],ROOT/spec['weights'])
        if 'torch' in sys.modules:
            raise RuntimeError('Approved submission unexpectedly imported torch')


class MatchEnv(OriginalMatchEnv):
    def __init__(self,own,foe,seat,approved_signatures):
        self.own_actor=NumpyActor(own,approved_signatures)
        self.foe_actor=NumpyActor(foe,approved_signatures)
        FairFightEnv.__init__(self,action_mode=self.own_actor.mode,seat=seat)


def duel(own,foe,band,n,approved_signatures):
    if n<=0 or n%2:
        raise ValueError('A match batch must contain complete seat pairs')
    rows=[];started=time.perf_counter()
    for seat in ('red','blue'):
        env=MatchEnv(own,foe,seat,approved_signatures)
        try:
            part=play(env,env.own_action,band,n//2,seat)
            for row in part: row['seat']=seat
            rows.extend(part)
        finally:
            env.close()
    return dict(own=own['id'],foe=foe['id'],band=band,summary=summary(rows),episodes=rows,
                elapsed_seconds=time.perf_counter()-started)
