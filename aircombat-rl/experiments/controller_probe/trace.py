"""Eight bounded follow-up traces: existing actor calls and physics only once."""
import argparse
from collections import Counter
import math
from pathlib import Path

from experiments.controller_probe.controllers import VARIANTS, make_controller
from experiments.controller_probe.run import verify_inputs, game_key
from experiments.rear_async.storage import read, write, sha


def run(out):
    plan=read(out/'plan.json');verify_inputs(plan)
    import experiments.aim_async.evaluate as metrics
    from tools.league_matches import MatchEnv
    from aircombat_gym.core.control.autopilot import wrap_pi
    baseline=read(out/'baseline.json');rows=[]
    for variant in VARIANTS:
        for seat,offset in (('red',3),('blue',0)):
            frames=[];calls=Counter()
            class TraceEnv(MatchEnv):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    ac=self._combat.ac[self.seat]
                    ac.autopilot=make_controller(variant,h0_ft=ac.h0_ft)
                    update=ac.autopilot.update
                    def observed(s,psi_cmd,*args,**kwargs):
                        calls['physics_updates']+=1
                        calls['near20_updates']+=abs(wrap_pi(psi_cmd-s.psi))<math.radians(20.)
                        return update(s,psi_cmd,*args,**kwargs)
                    ac.autopilot.update=observed

                def step(self,action):
                    dh,dv,_=self.decode(action)
                    calls['hold_heading_decisions']+=dh==0.
                    calls['policy_decisions']+=1
                    result=super().step(action)
                    ac=self._combat.ac[self.seat];s=ac.state;info=result[-1]
                    frames.append(dict(t=info['t'],delta_heading=dh,delta_speed=dv,
                        heading=s.psi,target=ac.psi_cmd,bank=s.phi,roll_rate=s.p,
                        aileron=ac.last.aileron,bank_target=ac.last.phi_cmd,
                        speed=s.v_kt,lead_error=info['ata_lead'],range=info['range'],
                        own_health=info['own_health'],opp_health=info['opp_health']))
                    return result
            previous=metrics.MatchEnv
            try:
                metrics.MatchEnv=TraceEnv
                foe=next(f for f in plan['opponents'] if f['id']=='lead')
                row=metrics.cell(plan['own_specs']['4900'],foe,plan['band']+offset,seat)
            finally:
                metrics.MatchEnv=previous
            job=dict(opponent=foe,seed=row['seed'],seat=seat,action_seed=4900)
            reference=baseline[game_key(job)]
            if variant=='original':
                differences={k:v for k,v in row.items() if k!='wall_seconds' and v!=reference[k]}
                if differences:raise ValueError('Read-only observation changed original: '+str(differences))
            item=dict(row,variant=variant,action_seed=4900,calls=dict(calls))
            rows.append(item)
            write(out/'combat_traces'/f'{variant}_{seat}.json',dict(summary=item,frames=frames))
            print(variant,seat,row['result'],dict(calls),flush=True)
    verify_inputs(plan)
    write(out/'trace_summary.json',dict(new_games=len(rows),training_steps=0,rows=rows,
        purpose='Two selected mutual-destruction lead fixtures; descriptive, not a new qualification panel.',
        original_metrics_exact=True,script_sha256=sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
