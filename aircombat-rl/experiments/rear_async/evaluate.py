"""Native FairFight metrics from same-step damage and current geometry."""
import time
from tools.league_matches import MatchEnv
from experiments.league.tail_reward import geometry
from experiments.league.health_rule_v2 import classify

def cell(own,foe,seed,seat):
    env=MatchEnv(own,foe,seat);start=time.perf_counter()
    try:
        obs,info=env.reset(seed=seed);hp,ehp=float(info['own_health']),float(info['opp_health'])
        rear=wez=fire=dealt_total=received_total=head=0.;streak=longest=0.;first_hit=first_rear=None;steps=0
        while True:
            obs,_,term,trunc,info=env.step(env.own_action(obs));steps+=1;g=geometry(obs)
            own,opp=float(info['own_health']),float(info['opp_health'])
            dealt=max(0.,ehp-opp);received=max(0.,hp-own)
            rear+=dealt*g['tail_attack'];wez+=.05*g['tail_attack'];fire+=.05*(g['tail_attack'] and dealt>0)
            streak=streak+.05 if g['tail_attack'] else 0.;longest=max(longest,streak)
            dealt_total+=dealt;received_total+=received;head+=received*g['head_on']
            if first_hit is None and received>0:first_hit=steps*.05
            if first_rear is None and dealt>0 and g['tail_attack']:first_rear=steps*.05
            hp,ehp=own,opp
            if term or trunc:break
        if not 0<=rear<=dealt_total+1e-9:raise ValueError('Rear damage measurement invalid')
        result=classify(hp,ehp)
        return dict(seed=seed,seat=seat,opponent=foe['id'],result=result,own_health=hp,opp_health=ehp,
            steps=steps,duration=float(env._combat.t),rear_damage=rear,rear_wez_seconds=wez,
            rear_damage_seconds=fire,max_rear_wez_streak=longest,damage_dealt=dealt_total,
            damage_received=received_total,head_on_damage=head,first_received_time=first_hit,
            first_rear_damage_time=first_rear,rear_before_first_received=first_rear is not None and (first_hit is None or first_rear<first_hit),
            clean_rear_win=result=='win' and rear>0 and received_total==0,wall_seconds=time.perf_counter()-start)
    finally:env.close()

def profile(rows):
    n=len(rows)
    if not n:raise ValueError('Empty panel')
    metrics=('rear_damage','rear_wez_seconds','max_rear_wez_streak','damage_dealt','damage_received','head_on_damage')
    return dict(games=n,wins=sum(r['result']=='win' for r in rows),win_rate=sum(r['result']=='win' for r in rows)/n,
        rear_damage_games=sum(r['rear_damage']>0 for r in rows),clean_rear_wins=sum(r['clean_rear_win'] for r in rows),
        rear_before_first_received_games=sum(r['rear_before_first_received'] for r in rows),
        **{k:sum(r[k] for r in rows)/n for k in metrics})
