"""Native single-aircraft tracking task with explicit information ablations."""
import math
import operator
import gymnasium as gym
import numpy as np
from aircombat_gym.core.aircraft import Aircraft
from aircombat_gym.core.control.autopilot import Autopilot,wrap_pi
from experiments.controller_probe.learnability import heading_rate
from experiments.controller_study.controllers import EffortController

VARIANTS=('original3','original7','effort7','stabilized7','effort7_obs','stabilized7_obs')
EFFORTS=(-1.,-.5,-.25,0.,.25,.5,1.)


class TrackingEnv(gym.Env):
    metadata={'render_modes':[]}
    render_mode=None

    def __init__(self,variant,stabilizer,horizon_steps=600):
        if variant not in VARIANTS:raise ValueError('Unknown tracking variant')
        if horizon_steps<=0:raise ValueError('Positive horizon required')
        self.variant=variant;self.horizon_steps=horizon_steps
        self.enriched=variant.endswith('_obs')
        self.original=variant.startswith('original')
        self.commands=(-30.,0.,30.) if variant=='original3' else (
            (-30.,-15.,-5.,0.,5.,15.,30.) if variant=='original7' else EFFORTS)
        settings=stabilizer if variant.startswith('stabilized') else {}
        self.ac=Aircraft(autopilot=Autopilot() if self.original else EffortController(**settings))
        self.action_space=gym.spaces.Discrete(len(self.commands))
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(19,),dtype=np.float32)

    def target(self,t):
        a=self.task
        angle=a['heading']+a['offset']+a['rate']*t+a['amplitude']*(
            math.sin(a['frequency']*t+a['phase'])-math.sin(a['phase']))
        rate=a['rate']+a['amplitude']*a['frequency']*math.cos(a['frequency']*t+a['phase'])
        return angle,rate

    def observation(self):
        s=self.ac.state;target,rate=self.target(self.steps*.05)
        error=wrap_pi(target-s.psi);c=self.ac.autopilot
        base=[math.sin(error),math.cos(error),rate/.3,s.v/340.,math.sin(s.phi),
            math.cos(s.phi),s.r/.5,s.nz/9.,(s.h_ft-20000.)/300.,s.h_dot/30.]
        extra=[s.p/2.,s.q,heading_rate(s)/.5,wrap_pi(self.ac.psi_cmd-s.psi)/math.pi,
            (self.ac.v_cmd_kt-s.v_kt)/100.,c._i_nz/2.5,c._i_v/60.,
            getattr(c,'filtered_phi',0.)/math.pi,self.last_command/30. if self.original else self.last_command]
        obs=np.asarray(base+(extra if self.enriched else [0.]*9),dtype=np.float32)
        if not np.isfinite(obs).all():raise RuntimeError('Nonfinite tracking observation')
        return obs

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        rng=self.np_random
        self.task=dict(speed=float(rng.choice([300.,450.,650.])),heading=float(rng.uniform(-math.pi,math.pi)),
            offset=float(rng.uniform(-math.pi/6,math.pi/6)),rate=float(rng.uniform(-math.radians(3),math.radians(3))),
            amplitude=float(rng.uniform(math.radians(5),math.radians(20))),
            frequency=float(2*math.pi/rng.uniform(12.,24.)),phase=float(rng.uniform(-math.pi,math.pi)))
        self.ac.reset(psi=self.task['heading'],v_kt=self.task['speed'])
        self.steps=0;self.last_command=0.;self.error_sum=0.;self.error_sq=0.;self.within=0
        self.altitude_max=0.;self.speed_error_sum=0.;self.return_sum=0.
        return self.observation(),dict(task=dict(self.task))

    def step(self,action):
        action=operator.index(action)
        if not self.action_space.contains(action):raise ValueError('Invalid discrete action')
        command=self.commands[action];self.last_command=command
        if self.original:self.ac.step(command)
        else:
            self.ac.autopilot.effort=command;self.ac.step(0.)
        self.steps+=1;s=self.ac.state
        if not s.flyable:raise RuntimeError('Tracking controller became unflyable')
        error=math.degrees(wrap_pi(self.target(self.steps*.05)[0]-s.psi))
        alt=abs(s.h_ft-20000.)
        reward=1.-min(abs(error)/45.,2.)-.1*min(alt/150.,1.)
        self.error_sum+=abs(error);self.error_sq+=error**2;self.within+=abs(error)<=5.
        self.altitude_max=max(self.altitude_max,alt)
        self.speed_error_sum+=abs(s.v_kt-self.task['speed']);self.return_sum+=reward
        truncated=self.steps>=self.horizon_steps
        info=dict(steps=self.steps,error_deg=error)
        if truncated:
            info['tracking']=dict(steps=self.steps,rmse_deg=math.sqrt(self.error_sq/self.steps),
                mae_deg=self.error_sum/self.steps,within5_fraction=self.within/self.steps,
                max_altitude_error_ft=self.altitude_max,mean_speed_error_kt=self.speed_error_sum/self.steps,
                episode_return=self.return_sum,task=dict(self.task))
        return self.observation(),reward,False,truncated,info

    def close(self):
        pass


def make_env(variant,stabilizer):
    return TrackingEnv(variant,stabilizer)
