"""Direction-invariant early-motion strategy selection; no opponent identity.

Two frozen flight policies remain reactive after selection. A small gate reads
the opponent's observed turn, speed change and closing motion during a learned
probe interval. Both the gate and its interval can be fitted by episodic search.
"""
from pathlib import Path
import json
import math
import zipfile
import numpy as np
from experiments.league.controller import Policy as Expert

GATE_NAMES=('probe_seconds','bias','mean_turn_weight','speed_change_weight',
            'heading_change_weight','closure_weight')
GATE_LOW=np.array([.5,-8,-8,-8,-8,-8.])
GATE_HIGH=np.array([6.,8,8,8,8,8.])
GATE_INITIAL=np.array([.5,1,-8,0,-8,0.])


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu',configuration=None):
        if configuration is None:
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as z:
                    configuration=json.loads(z.read('parameters.json'))
            else:
                configuration=json.loads(Path(weights).read_text(encoding='utf-8'))
        self.gate=np.asarray(configuration['gate'],float)
        if self.gate.shape!=GATE_LOW.shape or not np.isfinite(self.gate).all():
            raise ValueError('Expected six finite gate parameters')
        if np.any(self.gate<GATE_LOW) or np.any(self.gate>GATE_HIGH):
            raise ValueError('Gate parameters outside bounds')
        if len(configuration['experts'])!=2:
            raise ValueError('This gate selects exactly two frozen experts')
        self.experts=[Expert(parameters=e['parameters']) for e in configuration['experts']]
        self.reset()

    def reset(self):
        self.previous_remaining=None
        self.selected=None
        self.features=None
        self.turn_integral=0.
        self.probe_elapsed=0.

    def act(self,obs):
        x=obs
        remaining=float(x[38])
        # The public contract does not require the grader to call reset().
        # Detect a new episode using its clock, including after an early kill.
        if self.previous_remaining is not None and remaining>self.previous_remaining+1e-3:
            self.reset()
        distance=math.hypot(float(x[15]-x[0]),float(x[16]-x[1]))
        speed=math.sqrt(sum(float(v)**2 for v in x[18:21]))
        heading=float(x[26])
        turn=float(x[29])
        if self.previous_remaining is None:
            self.start_remaining=remaining
            self.start_distance=distance
            self.start_speed=speed
            self.start_heading=heading
            self.previous_turn=turn
        else:
            dt=max(0.,self.previous_remaining-remaining)
            self.turn_integral+=.5*(self.previous_turn+turn)*dt
            self.probe_elapsed+=dt
        self.previous_remaining=remaining
        self.previous_turn=turn
        if self.selected is None and self.probe_elapsed>=self.gate[0]-1e-5:
            elapsed=max(self.probe_elapsed,1e-6)
            delta_heading=math.atan2(math.sin(heading-self.start_heading),math.cos(heading-self.start_heading))
            self.features=np.array([1.,np.clip(abs(10*self.turn_integral/elapsed),0,3),
                np.clip((speed-self.start_speed)/20.,-3,3),
                np.clip(abs(delta_heading)/.35,0,3),np.clip((self.start_distance-distance)/elapsed/150.,-3,3)])
            self.selected=int(float(self.gate[1:]@self.features)>=0.)
        # Frozen expert zero performs the initial probe. No opponent label,
        # task side, simulator state, future action or private bot info is read.
        return self.experts[0 if self.selected is None else self.selected].act(obs)

    def __str__(self):
        return 'Direction-invariant two-expert policy with learned early gate'
