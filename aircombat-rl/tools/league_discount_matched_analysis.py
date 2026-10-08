"""Compare equal-budget discount pilots; action RNGs are not training repeats."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io


def describe(evaluations, rank, learner):
    ps=[e['profile'] for e in evaluations]
    ids={r['foe'] for r in evaluations[0]['results']}
    opponents={}
    for foe in sorted(ids):
        es=[x for e in evaluations for r in e['results'] if r['foe']==foe for x in r['episodes']]
        opponents[foe]=dict(games=len(es),wins=sum(x['outcome']=='kill' for x in es),
            deaths=sum(x['outcome']=='died' for x in es),mutual=sum(x['outcome']=='mutual' for x in es),
            timeouts=sum(x['outcome']=='timeout' for x in es),mean_end_seconds=sum(x['t'] for x in es)/len(es))
    return dict(win_rate=sum(p['mean_win_rate'] for p in ps)/len(ps),rank=rank,learner=learner,opponents=opponents)


def analyze(folder):
    plan=io.read(folder/'plan.json');io.verify(plan)
    raw=io.read(folder/'raw_discount_analysis.json');done=io.read(folder/'completion.json')
    assert raw['status']=='discount_raw_verified' and not plan['smoke']
    for name,digest in raw['input_sha256'].items():assert io.sha(io.ROOT/name)==digest,name
    initial=describe([io.read(folder/f'control/baseline_action_{a}.json') for a in plan['action_replicas']],done['baseline']['rank'],done['baseline']['learner'])
    rows=[]
    for seed in plan['seeds']:
        pair={}
        for arm in ('control','long_discount'):
            r=next(s for s in done['searches'] if s['arm']==arm and s['seed']==seed)
            assert r['additional_steps']==plan['chunk_steps'] and len(r['history'])==1
            c=r['history'][0]['candidate'];ck=io.ROOT/c['learner']
            pair[arm]=describe([io.read(ck.parent/f'development_{a}.json') for a in plan['action_replicas']],c['rank'],c['learner'])
        control,treatment=pair['control'],pair['long_discount']
        rows.append(dict(seed=seed,**pair,win_gain=treatment['win_rate']-control['win_rate'],
            gain_over_initial=treatment['win_rate']-initial['win_rate'],
            lower_quarter_gain=treatment['rank'][1]-control['rank'][1],
            worst_score_gain=treatment['rank'][2]-control['rank'][2],
            per_opponent_win_gain={k:treatment['opponents'][k]['wins']/treatment['opponents'][k]['games']-control['opponents'][k]['wins']/control['opponents'][k]['games'] for k in control['opponents']}))
    # A method-level extension requires both continuations to improve, not the
    # best seed or best action RNG. This screen is not heldout confirmation.
    promising=all(r['win_gain']>=.005-1e-12 and r['gain_over_initial']>=.005-1e-12
        and r['long_discount']['rank'][1]>=max(r['control']['rank'][1],initial['rank'][1])-1e-12
        and r['long_discount']['rank'][2]>=max(r['control']['rank'][2],initial['rank'][2])-1e-12 for r in rows)
    result=dict(status='matched_discount_comparison_verified',rows=rows,initial=initial,
        matched_continuations=len(rows),steps_per_pilot=plan['chunk_steps'],
        treatment_mean_win=sum(r['long_discount']['win_rate'] for r in rows)/len(rows),
        control_mean_win=sum(r['control']['win_rate'] for r in rows)/len(rows),
        mean_win_gain=sum(r['win_gain'] for r in rows)/len(rows),
        method_extension_screen_passed=promising,
        next_action='Freeze nominees for fresh-condition confirmation and independent initialization study before method adoption.' if promising else 'No automatic method-wide extension; preserve initial/best candidates and diagnose mutual-kill versus chase failures separately before a changed strategy.',
        rule='Both matched continuations must gain>=.005 win over control and common initial,with nondecreasing lower-quarter/worst scores versus both. Equal-budget final checkpoints,not retained fallbacks. This screen was implemented while pilots were running; no significance claim from two shared-ancestor repeats.',
        policy_promoted=False,scope=plan['quality_scope'],
        input_sha256={io.relative(p):io.sha(p) for p in [Path(__file__).resolve(),folder/'plan.json',folder/'completion.json',folder/'raw_discount_analysis.json']})
    io.write(folder/'matched_discount_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve())
    print(r['status'],{k:r[k] for k in ('treatment_mean_win','control_mean_win','mean_win_gain','method_extension_screen_passed')})
