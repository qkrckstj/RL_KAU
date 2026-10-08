"""Optional CUDA F-16 combat batches; never registered as the official Gym task.

The upstream physics port uses a local Earth approximation and interpolated trim.
Run the fidelity audit before using this experimental backend for research.
All hot-path state and outputs stay on the selected device. Match slots require
explicit reset; terminal observations remain available until that reset.
"""
import math
import torch

from .control import TensorAutopilot, wrap
from .geometry import engagement, observe

BACKEND_COMMIT = '1462017f6a5c724f9780c5d65b0c9552538ef426'


def score_step(health, track, time, inside, invalid):
    """Official flat-damage gun and four-outcome judge, plus explicit invalid code 4."""
    track = torch.where(inside,track+.05,0.)
    damage = torch.where(inside & (track >= 1.),.33*.05,0.)
    health = (health-damage.flip(-1)).clamp_min(0)
    time = time+.05
    red_dead,blue_dead = health[:,0] <= 0,health[:,1] <= 0
    mutual = (red_dead & (health[:,1] <= .02)) | (blue_dead & (health[:,0] <= .02))
    outcome = torch.where(time >= 120,3,0)
    outcome = torch.where(red_dead,-1,outcome)
    outcome = torch.where(blue_dead,1,outcome)
    outcome = torch.where(mutual,2,outcome)
    outcome = torch.where(invalid,4,outcome)
    return health,track,time,outcome != 0,outcome


class BatchedFairFight:
    """N games, two aircraft/game; actions (N,2) and raw observations (N,2,39).

outcome: 0 active, +1 red win, -1 blue win, 2 mutual, 3 timeout, 4 invalid.
Invalid physics is flagged and exposed as finite terminal data, never a win.
Callers must inspect `invalid` and reject training/evaluation runs containing it.
"""
    def __init__(self,n,device='cuda',dtype=torch.float32,seed=0):
        from jsbsim_f16_cuda import F16Stick,attach_stick
        if n < 1: raise ValueError('n must be positive')
        self.n,self.device,self.dtype = n,torch.device(device),dtype
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if dtype not in (torch.float32,torch.float64):
            raise ValueError('Physics requires float32 or float64')
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.dyn = F16Stick(2*n,device=device,dtype=dtype,lat0_deg=37.5665,refuel=True)
        self.fuel_initial = torch.tensor([1500.,1500.,0.,0.],device=device,dtype=dtype).expand(2*n,4).clone()
        if self.device.type == 'cuda': attach_stick(self.dyn)
        self.control = TensorAutopilot(2*n,device,dtype)
        self.heading = torch.zeros(2*n,device=device,dtype=dtype)
        self.speed = torch.zeros_like(self.heading)
        # Python's original clock/health bookkeeping uses doubles even when observations are float32.
        self.health = torch.ones(n,2,device=device,dtype=torch.float64)
        self.track = torch.zeros_like(self.health)
        self.time = torch.zeros(n,device=device,dtype=torch.float64)
        self.done = torch.zeros(n,device=device,dtype=torch.bool)
        self.invalid = torch.zeros_like(self.done)
        self.outcome = torch.zeros(n,device=device,dtype=torch.int64)
        self.initial = torch.zeros(n,2,4,device=device,dtype=dtype)
        self.raw = torch.zeros(n,2,15,device=device,dtype=dtype)
        self.obs = torch.zeros(n,2,39,device=device,dtype=dtype)
        self.actions = torch.full((n,2),4,device=device,dtype=torch.int64)
        self.graph = None
        self.reset_graph = None
        self.reset_mask = torch.zeros_like(self.done)
        self.reset_initial = torch.zeros_like(self.initial)
        self.reset()
        self._safe_state = [x.clone() for x in self.dyn.watched_tensors().values()]

    def _aircraft(self):
        from jsbsim_f16_cuda.rbdyn import euler_from_quat
        d = self.dyn
        pos,vel = d.rb.pos_ned,d.rb.velocity_ned()
        phi,theta,psi = euler_from_quat(d.rb.quat)
        return torch.stack((pos[:,1],pos[:,0],-pos[:,2],vel[:,1],vel[:,0],-vel[:,2],
            d.n_pilot[:,0],d.n_pilot[:,1],-d.n_pilot[:,2],phi,theta,psi,
            d.rb.pqr[:,0],d.rb.pqr[:,1],d.rb.pqr[:,2]),-1).reshape(self.n,2,15)

    @torch.no_grad()
    def reset(self,mask=None,initial=None):
        """Initial fields are (east m, north m, heading rad, speed kt), both seats.

Default sampling has the official distribution, using a Torch RNG. For exact
paired initial conditions pass samples from FairFightEnv.sample explicitly.
"""
        mask = torch.ones_like(self.done) if mask is None else mask.clone()
        if mask.shape != (self.n,) or mask.device != self.done.device or mask.dtype != torch.bool:
            raise ValueError('reset mask must be an (N,) bool tensor on the environment device')
        if initial is None:
            initial = self.sample_initial()
        else:
            if initial.shape != (self.n,2,4): raise ValueError('Expected initial shape (N,2,4)')
            initial = initial.to(device=self.device,dtype=self.dtype)
        self.initial.copy_(torch.where(mask[:,None,None],initial,self.initial))
        ic = initial.reshape(-1,4)
        pos = torch.stack((ic[:,1],ic[:,0],torch.full_like(ic[:,0],-6096)),-1)
        m = mask.repeat_interleave(2)
        old_n2norm=self.dyn.n2norm.clone()
        ok = self.dyn.reset(pos,ic[:,2],ic[:,3]*.514444,fuel_lbs=self.fuel_initial,mask=m)
        # Upstream recomputes this derived value for all rows during masked reset;
        # preserve non-reset rows bit for bit (CUDA and Torch round differently).
        self.dyn.n2norm.copy_(torch.where(m,self.dyn.n2norm,old_n2norm))
        # Original backend overwrites trim controls with neutral stick / 0.5 throttle.
        neutral=torch.zeros_like(self.dyn.stick);neutral[:,3]=.5
        self.dyn.stick.copy_(torch.where(m[:,None],neutral,self.dyn.stick))
        self.heading.copy_(torch.where(m,ic[:,2],self.heading))
        self.speed.copy_(torch.where(m,ic[:,3],self.speed))
        self.control.i_nz.masked_fill_(m,0)
        self.control.i_v.masked_fill_(m,0)
        self.health.masked_fill_(mask[:,None],1)
        self.track.masked_fill_(mask[:,None],0)
        self.time.masked_fill_(mask,0)
        bad=(~ok.reshape(self.n,2)).any(-1)
        self.invalid.copy_(torch.where(mask,bad,self.invalid))
        self.done.copy_(torch.where(mask,bad,self.done))
        self.outcome.copy_(torch.where(mask,bad.long()*4,self.outcome))
        self.raw.copy_(self._aircraft())
        updated=observe(self.raw,self.health.to(self.dtype),self.track.to(self.dtype),
                        engagement(self.raw)['in_wez'],(120-self.time).to(self.dtype))
        self.obs.copy_(torch.where(mask[:,None,None],updated,self.obs))
        return self.obs

    def sample_initial(self):
        uniform=torch.rand(self.n,3,device=self.device,dtype=self.dtype,generator=self.rng)
        initial=torch.zeros_like(self.initial)
        initial[:,0,0],initial[:,1,0]=-5000,5000
        initial[:,:,2]=(uniform[:,1:]*20-10)*math.pi/180
        initial[:,1,2]+=math.pi
        initial[:,:,3]=(405+90*uniform[:,0,None]).expand(-1,2)
        return initial

    @torch.no_grad()
    def reset_done(self):
        """Reset ended slots without CPU synchronization, retaining other episodes."""
        if self.reset_graph is None: return self.reset(mask=self.done)
        self.reset_mask.copy_(self.done)
        self.reset_initial.copy_(self.sample_initial())
        self.reset_graph.replay()
        return self.obs

    def _buffers(self):
        return list(self.dyn.watched_tensors().values()) + [self.control.i_nz,self.control.i_v,
            self.heading,self.speed,self.health,self.track,self.time,self.done,self.invalid,
            self.outcome,self.initial,self.raw,self.obs,self.actions]

    def _step(self):
        active = ~self.done
        flying = active.repeat_interleave(2)
        watched = list(self.dyn.watched_tensors().values())+[self.control.i_nz,self.control.i_v]
        before = [x.clone() for x in watched]
        action = self.actions.reshape(-1)
        bad_action = ((self.actions < 0)|(self.actions > 8)).any(-1)
        dh = (action//3-1)*30
        dv = (action%3-1)*20
        raw=self.raw.reshape(-1,15)
        self.heading.copy_(torch.where((dh != 0)&flying,wrap(raw[:,11]+dh*math.pi/180),self.heading))
        self.speed.copy_(torch.where((dv != 0)&flying,
            (torch.linalg.vector_norm(raw[:,3:6],dim=-1)/.514444+dv).clamp(300,650),self.speed))
        for _ in range(6):
            aircraft=self._aircraft().reshape(-1,15)
            bad=~torch.isfinite(aircraft).all(-1)
            self.invalid.logical_or_(bad.reshape(self.n,2).any(-1)&active)
            # A NaN index in a lookup table can poison the entire CUDA context.
            # Quarantine invalid rows before either control or physics lookup;
            # the match remains flagged invalid and cannot count as a win.
            for buf,safe in zip(self.dyn.watched_tensors().values(),self._safe_state):
                buf.copy_(torch.where(bad.reshape((-1,)+(1,)*(buf.ndim-1)),safe,buf))
            aircraft=torch.where(bad[:,None],self.raw.reshape(-1,15),aircraft)
            controls = self.control.update(aircraft,self.heading,self.speed)
            self.dyn.step(controls,substeps=1)
        for buf,old in zip(watched,before):
            mask=flying.reshape((-1,)+(1,)*(buf.ndim-1))
            buf.copy_(torch.where(mask,buf,old))
        raw=self._aircraft()
        finite=torch.isfinite(raw).all(dim=(-1,-2))
        valid_height=((raw[:,:,2] > 0)&(raw[:,:,2] < 30480)).all(-1)
        invalid=(~finite)|(~valid_height)|bad_action
        self.invalid.logical_or_(invalid & active)
        raw=torch.where(invalid[:,None,None],self.raw,raw)
        self.raw.copy_(raw)
        inside=engagement(raw)['in_wez']
        old_track=self.track.clone()
        health,track,time,done,outcome=score_step(self.health,self.track,self.time,inside,self.invalid)
        self.health.copy_(torch.where(active[:,None],health,self.health))
        self.track.copy_(torch.where(active[:,None],track,self.track))
        self.time.copy_(torch.where(active,time,self.time))
        self.outcome.copy_(torch.where(active,outcome,self.outcome))
        self.done.logical_or_(done)
        # Official Combat patches health after shooting, but exposes the pre-update track counter.
        obs=observe(raw,self.health.to(self.dtype),old_track.to(self.dtype),inside,(120-self.time).to(self.dtype))
        self.obs.copy_(torch.where(active[:,None,None],obs,self.obs))

    @torch.no_grad()
    def step(self,actions):
        if actions.shape != (self.n,2) or actions.dtype != torch.int64 or actions.device != self.actions.device:
            raise ValueError('actions must be (N,2) int64 on the environment device')
        self.actions.copy_(actions)
        if self.graph is None: self._step()
        else: self.graph.replay()
        return self.obs,self.done,self.outcome

    @torch.no_grad()
    def capture(self):
        """Capture one complete combat decision, retaining all original state."""
        if self.device.type != 'cuda': raise ValueError('CUDA graph requires CUDA')
        if self.graph is not None: return
        buffers=self._buffers()
        saved=[x.clone() for x in buffers]
        side=torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2): self._step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self._step()
        self.reset_initial.copy_(self.initial)
        self.reset_mask.zero_()
        with torch.cuda.stream(side):
            self.reset(mask=self.reset_mask,initial=self.reset_initial)
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.reset_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.reset_graph):
            self.reset(mask=self.reset_mask,initial=self.reset_initial)
        for buf,value in zip(buffers,saved): buf.copy_(value)
