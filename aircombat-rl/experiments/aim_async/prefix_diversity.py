"""Training-only diversity of native first-pass sources; inference is unchanged."""
import copy
import math
import numpy as np

from experiments.aim_async.post_encounter import PostEncounterArchive, PrefixRestartable
from tools.league_matches import Actor


class PrefixPoolActor:
    def __init__(self, config):
        if set(config) != {'salt','weights','sources'} or type(config['salt']) is not int or config['salt']<0:
            raise ValueError('Expected a nonnegative independent selector salt')
        if len(config['sources'])!=2 or len(config['weights'])!=2:
            raise ValueError('Expected two frozen prefix sources')
        weights=np.asarray(config['weights'],dtype=float)
        if not np.isfinite(weights).all() or np.any(weights<0) or not math.isclose(float(weights.sum()),1.,abs_tol=1e-12):
            raise ValueError('Expected normalized source probabilities')
        if any(set(s)!={'role','spec'} for s in config['sources']) or [s['role'] for s in config['sources']]!=['parent','candidate']:
            raise ValueError('Expected ordered parent and candidate sources')
        self.config=copy.deepcopy(config)
        self.actors=[Actor(s['spec']) for s in config['sources']]
        if any(a.mode!='discrete' for a in self.actors):raise ValueError('Discrete prefix sources required')
        self.mode='discrete';self.selected=None

    def seed(self, seed, physical_seat):
        selector=np.random.default_rng(np.random.SeedSequence([int(seed),self.config['salt']]))
        self.selected=int(selector.choice(2,p=self.config['weights']))
        self.actors[self.selected].seed(seed,physical_seat)

    @property
    def role(self):
        if self.selected is None:raise RuntimeError('Select a prefix before using it')
        return self.config['sources'][self.selected]['role']

    def reset(self):
        if self.selected is None:raise RuntimeError('Select a prefix before reset')
        self.actors[self.selected].reset()

    def act(self, info=None, obs=None):
        if self.selected is None:raise RuntimeError('Select a prefix before action')
        return self.actors[self.selected].act(info,obs)


class DiversePrefixArchive(PostEncounterArchive):
    def __init__(self,*args,prefix_pool,prefix_spec,**kwargs):
        if prefix_pool['sources'][0]['spec']!=prefix_spec:
            raise ValueError('Original parent prefix must be the first pool source')
        super().__init__(*args,prefix_spec=prefix_spec,**kwargs)
        self.prefix_pool=copy.deepcopy(prefix_pool)
        self.prefix_actor=PrefixPoolActor(prefix_pool)
        self.prefix_source_totals={role:dict(attempted=0,post_encounter=0,fallback_no_entry=0,
            fallback_no_exit=0,native_steps=0,native_seconds=0.) for role in ('parent','candidate')}

    def reset(self,**kwargs):
        raw,info=super().reset(**kwargs)
        prefix=self.episode_metadata['prefix']
        role=self.prefix_actor.role if prefix['attempted'] else None
        prefix['source_role']=role
        if role is not None:
            total=self.prefix_source_totals[role]
            total['attempted']+=1;total[prefix['status']]+=1
            total['native_steps']+=prefix['native_steps'];total['native_seconds']+=prefix['native_seconds']
        return raw,dict(info,league_episode=copy.deepcopy(self.episode_metadata))


class DiversePrefixRestartable(PrefixRestartable):
    def snapshot_worker(self):
        return dict(super().snapshot_worker(),prefix_pool=copy.deepcopy(self.unwrapped.prefix_pool),
                    prefix_source_totals=copy.deepcopy(self.unwrapped.prefix_source_totals))

    def restore_worker(self,state):
        if state.get('prefix_pool')!=self.unwrapped.prefix_pool or 'prefix_source_totals' not in state:
            raise ValueError('Saved prefix source configuration/accounting missing or changed')
        totals=state['prefix_source_totals']
        if totals.keys()!=self.unwrapped.prefix_source_totals.keys():raise ValueError('Saved prefix sources differ')
        for role,values in totals.items():
            if values.keys()!=self.unwrapped.prefix_source_totals[role].keys() or any(not math.isfinite(v) or v<0 for v in values.values()):
                raise ValueError('Invalid saved prefix accounting')
        super().restore_worker(state)
        self.unwrapped.prefix_source_totals=copy.deepcopy(totals)


def make_worker(plan,index,restart=None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from experiments.aim_async.draw_utility_worker import DrawUtilityGuidanceEnv
    torch.set_num_threads(1)
    archive=DiversePrefixArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],
        curriculum=plan['curriculum'],prefix_spec=plan['prefix_spec'],prefix_pool=plan['prefix_pool'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=DrawUtilityGuidanceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],
        plan['control_guidance_range'],draw_penalty=plan['terminal_draw_penalty'])
    env=DiversePrefixRestartable(Monitor(wrapped,info_keywords=('verdict_rule','result','terminal_reason',
        'damage_dealt','damage_received','connected_fire_seconds','lock_breaks',
        'first_damage_time','first_received_time','first_shot_advantage',
        'relative_shaping_total','control_guidance_total','draw_utility_total')))
    if restart is not None:env.restore_worker(restart)
    return env
