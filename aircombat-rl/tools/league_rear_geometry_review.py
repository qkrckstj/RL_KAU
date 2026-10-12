"""Independent completed-run accounting and paired development comparison."""
from pathlib import Path
import json,math
from tools import league_thread_benchmark as io
from experiments.league.health_rule_v2 import classify


def run():
    root=io.ROOT/'runs/league_rear_geometry_20261012'
    p=io.read(root/'evaluation_plan.json');io.verify(p);d=io.read(root/'completion.json')
    assert d['status']=='rear_geometry_complete'
    train=io.read(root/'training_results.json');training={}
    for role,r in train.items():
        es=[json.loads(x) for x in (root/role/'training_episodes.jsonl').read_text('utf-8').splitlines()]
        assert len(es)==r['training_episodes']
        assert r['steps']==655360 and r['optimizer_restore_exact']
        assert p['tail_coefficients']['received']==.5
        for e in es:
            assert e['result']==classify(e['own_health'],e['opp_health'])
            assert 0<=e['damage_received']<=1.+1e-9
            wanted='quarter_focus_ids' if e['curriculum_geometry']=='training_tail_quarter' else 'focus_ids'
            assert e['opponent_focus_ids']==p['opponent_schedule'][wanted]
            assert abs(e['opponent_focus_weight']-max(0.,1.-e['schedule_start_step']/10240))<1e-12
        band=p['training_bands'][p['seeds'].index(r['seed'])]
        streams={i:[] for i in range(32)}
        for e in es:
            worker=(e['episode_seed']-band)%32
            assert e['episode_seed']==band+e['episode_index']*32+worker
            streams[worker].append(e)
        for stream in streams.values():
            assert stream
            stream.sort(key=lambda e:e['episode_index'])
            assert stream[0]['schedule_start_step']==0
            for a,b in zip(stream,stream[1:]):
                assert b['episode_index']==a['episode_index']+1
                assert b['schedule_start_step']==a['schedule_start_step']+a['l']
        training[role]=dict(games=len(es),rear_damage_games=sum(e['actual_rear_damage']>0 for e in es),
            opponents_seen=len({e['opponent_id'] for e in es}),
            completed_original_weight_games=sum(e['opponent_focus_weight']==0 for e in es),
            physical_step_streams_verified=32,scope='Mixture training starts, not FairFight performance evidence')
    cells={role:{} for role in p['evaluation_roles']}
    for file in (root/'evaluation').glob('match_*.json'):
        b=io.read(file);rep=int(b['own'].rsplit('_a',1)[1])
        for e in b['episodes']:
            key=(b['foe'],e['seed'],e['seat'],rep)
            assert key not in cells[b['role']];cells[b['role']][key]=e
            assert e['result']==classify(e['own_health'],e['opp_health'])
    expected={(f['id'],115000000+k,seat,a) for f in p['development_opponents'] for k in (0,1) for seat in ('red','blue') for a in (4900,4901)}
    assert len(expected)==192 and all(set(c)==expected for c in cells.values())
    effects={}
    metrics=('win_rate','rear_damage','rear_wez_seconds','damage_dealt','damage_received','head_on_damage_current')
    val=lambda e,m:float(e['result']=='win') if m=='win_rate' else e[m]
    for seed in p['seeds']:
        left,right=cells[f'baseline_s{seed}'],cells[f'geometry_s{seed}'];parts={}
        for scope in ['all','noncircler']+[f['id'] for f in p['development_opponents']]:
            keys=[k for k in expected if scope=='all' or (scope=='noncircler' and k[0]!='circler') or k[0]==scope]
            part={}
            for m in metrics:
                a=sum(val(left[k],m) for k in keys)/len(keys);b=sum(val(right[k],m) for k in keys)/len(keys)
                part[m]=dict(baseline=a,candidate=b,delta=b-a)
            part['games']=len(keys);part['rear_damage_games']={'baseline':sum(left[k]['rear_damage']>0 for k in keys),'candidate':sum(right[k]['rear_damage']>0 for k in keys)}
            parts[scope]=part
        effects[str(seed)]=parts
    for role,rows in cells.items():
        profile=d['profiles'][role]['all'];assert profile['games']==192 and profile['wins']==sum(e['result']=='win' for e in rows.values())
        for m in metrics[1:]:assert abs(profile[m]-sum(e[m] for e in rows.values())/192)<1e-12
    eligible=[seed for seed,parts in effects.items() if parts['all']['win_rate']['delta']>=-.02 and parts['all']['damage_received']['delta']<=.02 and parts['noncircler']['rear_damage']['delta']>1e-6]
    decision='candidate_for_further_confirmation' if len(eligible)==2 else 'no_consistent_transfer_do_not_extend_same_setting'
    report=dict(status='paired_geometry_review_complete',training=training,effects=effects,eligible_seeds=eligible,decision=decision,scope='Two shared-ancestor matched continuations, consumed development IC. No unseen final confirmation/statistical superiority claim.',policy_promoted=False)
    io.write(root/'review.json',report);print(dict(decision=decision,effects={seed:{k:v for k,v in part.items() if k in ('all','noncircler')} for seed,part in effects.items()}),flush=True)

if __name__=='__main__':run()
