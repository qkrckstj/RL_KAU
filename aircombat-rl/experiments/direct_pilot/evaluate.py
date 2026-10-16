"""Unchanged native metrics plus pre-action public threat gate accounting."""
import math
import time
from tools.league_matches import MatchEnv
from aircombat_gym.wvr import obs as O
from experiments.aim_async.threat_metrics import ThreatCounter
from experiments.aim_async.wez_or_damaged_metrics import WezOrDamagedCounter
from experiments.aim_async.early_range_metrics import EarlyRangeCounter,early_range_gate_from_raw
from experiments.aim_async.range_metrics import RangeCounter,range_gate_from_raw
from experiments.league.tail_reward import geometry
from experiments.league.health_rule_v2 import classify
from experiments.rear_async.evaluate import profile as combat_profile


def cell(own,foe,seed,seat):
    env=MatchEnv(own,foe,seat);start=time.perf_counter()
    try:
        obs,info=env.reset(seed=seed);hp,ehp=float(info['own_health']),float(info['opp_health'])
        counter=ThreatCounter();combined_counter=WezOrDamagedCounter();early_counter=EarlyRangeCounter();range_counter=RangeCounter()
        rear=wez=fire=dealt_total=received_total=head=0.;streak=longest=0.
        first_hit=first_rear=first_dealt=None;steps=0
        unilateral_streak=longest_unilateral=posthit_rear_streak=longest_posthit_rear=0.
        first_posthit_unilateral=received_at_first_posthit=None
        longest_light_posthit_rear=0.
        connected=locked_seconds=0.;lock_breaks=0;was_locked=False
        lead_sum=0.;range_samples=0
        while True:
            gate=bool(obs[O.index('opp_in_wez')]>.5)
            combined_gate=gate or bool(obs[O.index('own_health')]<1.)
            early_gate=early_range_gate_from_raw(obs)
            expanded_gate=range_gate_from_raw(obs,8000.)
            obs,_,term,trunc,info=env.step(env.own_action(obs));steps+=1;g=geometry(obs)
            counter.observe(gate);combined_counter.observe(combined_gate);early_counter.observe(early_gate);range_counter.observe(expanded_gate)
            own_hp,opp_hp=float(info['own_health']),float(info['opp_health'])
            dealt=max(0.,ehp-opp_hp);received=max(0.,hp-own_hp)
            rear+=dealt*g['tail_attack'];wez+=.05*g['tail_attack'];fire+=.05*(g['tail_attack'] and dealt>0)
            streak=streak+.05 if g['tail_attack'] else 0.;longest=max(longest,streak)
            dealt_total+=dealt;received_total+=received;head+=received*g['head_on']
            now=float(info['t'])
            if first_hit is None and received>0:first_hit=now
            if first_dealt is None and dealt>0:first_dealt=now
            if first_rear is None and dealt>0 and g['tail_attack']:first_rear=now
            unilateral=bool(obs[34]>.5 and obs[35]<=.5)
            unilateral_streak=unilateral_streak+.05 if unilateral else 0.
            longest_unilateral=max(longest_unilateral,unilateral_streak)
            after_hit=first_hit is not None and now>first_hit
            if after_hit and unilateral and first_posthit_unilateral is None:
                first_posthit_unilateral=now;received_at_first_posthit=received_total
            posthit_rear_streak=posthit_rear_streak+.05 if after_hit and unilateral and g['tail_attack'] else 0.
            longest_posthit_rear=max(longest_posthit_rear,posthit_rear_streak)
            if received_total<=.25:
                longest_light_posthit_rear=max(longest_light_posthit_rear,posthit_rear_streak)
            connected+=.05*(dealt>0)
            locked=bool(info['in_wez'] and info['track_time']>=env.track_lock)
            locked_seconds+=.05*locked;lock_breaks+=int(was_locked and not locked);was_locked=locked
            if env.wez_r_min<=info['range']<=env.wez_r_max:
                lead_sum+=math.degrees(info['ata_lead']);range_samples+=1
            hp,ehp=own_hp,opp_hp
            if term or trunc:break
        result=classify(hp,ehp)
        return dict(**counter.record(),**combined_counter.record(),**early_counter.record(),**range_counter.record(),seed=seed,seat=seat,opponent=foe['id'],result=result,own_health=hp,opp_health=ehp,
            max_light_posthit_rear_unilateral_streak=longest_light_posthit_rear,
            max_unilateral_wez_streak=longest_unilateral,max_posthit_rear_unilateral_streak=longest_posthit_rear,
            first_posthit_unilateral_time=first_posthit_unilateral,damage_received_before_first_posthit_unilateral=received_at_first_posthit,
            steps=steps,duration=float(env._combat.t),rear_damage=rear,rear_wez_seconds=wez,
            rear_damage_seconds=fire,max_rear_wez_streak=longest,damage_dealt=dealt_total,
            damage_received=received_total,head_on_damage=head,first_received_time=first_hit,
            first_rear_damage_time=first_rear,rear_before_first_received=first_rear is not None and(first_hit is None or first_rear<first_hit),
            clean_rear_win=result=='win' and rear>0 and received_total==0,
            first_damage_time=first_dealt,first_shot_advantage=first_dealt is not None and(first_hit is None or first_dealt<first_hit),
            # No-hit censoring uses the common horizon, not a misleading hit-only average.
            first_damage_time_censored=env.t_max if first_dealt is None else first_dealt,
            connected_fire_seconds=connected,locked_fire_seconds=locked_seconds,lock_breaks=lock_breaks,
            in_range_lead_error_deg=None if not range_samples else lead_sum/range_samples,
            in_range_samples=range_samples,wall_seconds=time.perf_counter()-start)
    finally:env.close()


def profile(rows):
    result=combat_profile(rows)
    aiming=[r for r in rows if 'connected_fire_seconds' in r]
    if aiming and len(aiming)!=len(rows):raise ValueError('Mixed aiming-metric coverage')
    if aiming:
        n=len(rows)
        for name in ('connected_fire_seconds','locked_fire_seconds','lock_breaks','first_damage_time_censored'):
            result[name]=sum(r[name] for r in rows)/n
        result['first_shot_advantage_rate']=sum(r['first_shot_advantage'] for r in rows)/n
        ranged=[r for r in rows if r['in_range_lead_error_deg'] is not None]
        result['in_range_lead_error_deg']=None if not ranged else sum(r['in_range_lead_error_deg'] for r in ranged)/len(ranged)
        result['aiming_metric_games']=n
    return result
