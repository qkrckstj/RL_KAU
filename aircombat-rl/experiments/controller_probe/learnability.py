"""Throwaway environment-interface probe, not a policy or installed controller.

Turn effort requests a fraction of the nominal available level-turn rate.
Feedforward is approximate; actual rate must be measured in native JSBSim.
"""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import time

from aircombat_gym.core.control.autopilot import Autopilot, wrap_pi
from aircombat_gym.core.aircraft import Aircraft
from experiments.rear_async.storage import read, write, sha

ROOT=Path(__file__).resolve().parents[2]


class TurnEffortAutopilot(Autopilot):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.effort=0.

    def reset(self):
        super().reset()
        self.effort=0.

    def update(self,s,psi_cmd,v_cmd_kt,dt,alt_cmd_ft=None):
        if not math.isfinite(self.effort) or abs(self.effort)>1.:
            raise ValueError('Turn effort must be finite and in [-1,1]')
        out=super().update(s,psi_cmd,v_cmd_kt,dt,alt_cmd_ft)
        phi=math.atan(self.effort*math.tan(out.max_bank))
        aileron=max(-1.,min(1.,self.g.k_phi*wrap_pi(phi-s.phi)-self.g.k_p*s.p))
        return replace(out,phi_cmd=phi,aileron=aileron)


def heading_rate(s):
    cosine=math.cos(s.theta)
    return (s.q*math.sin(s.phi)+s.r*math.cos(s.phi))/math.copysign(max(abs(cosine),1e-6),cosine)


def trajectory(kind,speed,command,release=False):
    ac=Aircraft(autopilot=Autopilot() if kind=='original' else TurnEffortAutopilot())
    initial=ac.reset(v_kt=speed)
    frames=[]
    for i in range(360 if release else 300):
        u=0. if release and i>=80 else command
        if kind=='original':
            ac.step(u)
        else:
            ac.autopilot.effort=u
            ac.step(0.)
        s=ac.state
        if not s.flyable:raise RuntimeError('Unflyable probe')
        frames.append(dict(t=(i+1)*.05,heading=s.psi,bank=s.phi,
            heading_rate_deg=math.degrees(heading_rate(s)),
            nominal_max_rate_deg=math.degrees(9.80665*math.tan(ac.last.max_bank)/s.v),
            command=u,bank_target=ac.last.phi_cmd,aileron=ac.last.aileron,
            altitude_ft=s.h_ft,speed_kt=s.v_kt))
    tail=frames[-80:]
    stop=None
    if release:
        for i in range(80,len(frames)-20):
            if all(abs(f['heading_rate_deg'])<.5 for f in frames[i:]):
                stop=frames[i]['t']-4.;break
    return dict(kind=kind,speed=speed,command=command,release=release,
        initial={k:float(v).hex() for k,v in vars(initial).items()},frames=frames,
        metrics=dict(mean_final_rate_deg=sum(f['heading_rate_deg'] for f in tail)/len(tail),
            mean_final_nominal_rate_deg=sum(f['nominal_max_rate_deg'] for f in tail)/len(tail),
            max_altitude_error_ft=max(abs(f['altitude_ft']-20000.) for f in frames),
            max_speed_error_kt=max(abs(f['speed_kt']-speed) for f in frames),
            stop_within_half_deg_per_second_after_release=stop,
            residual_turn_after_release_deg=math.degrees(wrap_pi(frames[-1]['heading']-frames[79]['heading'])) if release else None))


def hidden_state_probe():
    ac=Aircraft();s=ac.reset(v_kt=450.)
    results=[]
    for label,target,i_nz,i_v,vtarget in (
        ('heading_left',-.5,0.,0.,450.),('heading_right',.5,0.,0.,450.),
        ('pitch_memory_minus',0.,-.5,0.,450.),('pitch_memory_plus',0.,.5,0.,450.),
        ('speed_memory_slow',0.,0.,0.,430.),('speed_memory_fast',0.,0.,0.,470.)):
        controller=Autopilot();controller._i_nz=i_nz;controller._i_v=i_v
        out=controller.update(s,target,vtarget,1/120)
        results.append(dict(label=label,heading_target=target,stored_i_nz=i_nz,
            speed_target=vtarget,aileron=out.aileron,elevator=out.elevator,throttle=out.throttle))
    return dict(scope='Synthetic controller-state counterexample using one identical physical state. '
        'Not a claim that these full states co-occur in reachable match trajectories.',results=results)


def main(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source_files=list((ROOT/'aircombat_gym').rglob('*.py'))+[
        Path(__file__),ROOT/'experiments/rear_async/storage.py',ROOT/'tests/test_controller_learnability.py']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    write(out/'plan.json',dict(purpose='Environment controllability and action semantics, not frozen PPO wins',
        nominal_efforts=[0.,.1,.25,.5,1.],original_commands_deg=[0.,1.,2.,5.,15.,30.],
        speeds_kt=[300.,450.,650.],steps_of_training=0,combat_games=0,source_sha256=frozen))
    start=time.perf_counter();results=[];initials={}
    for speed in (300.,450.,650.):
        for kind,commands in (('original',(0.,1.,2.,5.,15.,30.)),('turn_effort',(0.,.1,.25,.5,1.))):
            for command in commands:
                trace=trajectory(kind,speed,command)
                name=f'{kind}_{int(speed)}_{command:g}'
                write(out/'traces'/(name+'.json'),trace)
                if speed in initials and trace['initial']!=initials[speed]:raise ValueError('Initial states differ')
                initials[speed]=trace['initial']
                results.append({k:v for k,v in trace.items() if k not in ('frames','initial')})
                print(name,trace['metrics']['mean_final_rate_deg'],flush=True)
            trace=trajectory(kind,speed,30. if kind=='original' else 1.,release=True)
            write(out/'traces'/f'{kind}_{int(speed)}_release.json',trace)
            if trace['initial']!=initials[speed]:raise ValueError('Release initial differs')
            results.append({k:v for k,v in trace.items() if k not in ('frames','initial')})
    aliases={}
    for speed in (300.,450.,650.):
        traces={c:read(out/'traces'/f'original_{int(speed)}_{c}.json')['frames'] for c in (5,15,30)}
        # Exclude the deliberately different command fields; compare actual
        # trajectory and controller outputs at every decision step.
        physical=lambda fs:[{k:v for k,v in f.items() if k!='command'} for f in fs]
        aliases[str(int(speed))]=dict(five_equals_fifteen=physical(traces[5])==physical(traces[15]),
            fifteen_equals_thirty=physical(traces[15])==physical(traces[30]))
    for name,digest in frozen.items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed during probe: '+name)
    write(out/'results.json',dict(status='complete',rows=results,aliasing=aliases,
        hidden_state=hidden_state_probe(),initial_states_exact=True,
        trajectories=len(results),training_steps=0,combat_games=0,
        wall_seconds=time.perf_counter()-start,installed=False,
        limitation='This measures command response, not PPO learning speed or final combat quality.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
