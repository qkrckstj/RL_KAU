"""Compare matched entropy treatments without mistaking action RNGs for learners."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io


def verified(folder):
    plan = io.read(folder/'plan.json')
    io.verify(plan)
    raw = io.read(folder/'raw_extend_analysis.json')
    assert raw['status'] == 'extend_teacher_raw_verified'
    for name, digest in raw['input_sha256'].items():
        assert io.sha(io.ROOT/name) == digest, name
    return plan, raw, io.read(folder/'completion.json')


def analyze(folder):
    plan, raw, done = verified(folder)
    source = io.ROOT/plan['matched_control_source']
    control, control_raw, control_done = verified(source)
    assert not plan['smoke']
    for key in ['teacher','opponents','groups','probabilities','coverage_period',
                'development_opponents','development_band','development_n','training_bands',
                'seeds','chunk_steps','action_replicas','workers','processes']:
        assert plan[key] == control[key], key
    a,b = dict(plan['ppo']),dict(control['ppo'])
    assert a.pop('ent_coef') == 0. and b.pop('ent_coef') == .005 and a == b
    rows=[]
    for seed in plan['seeds']:
        for key in ['parameter_digest','optimizer_has_no_history','source_steps']:
            assert plan['initializations']['prior6'][str(seed)][key] == control['initializations']['prior6'][str(seed)][key]
        pair=[]
        for root, completed in [(source,control_done),(folder,done)]:
            search=next(r for r in completed['searches'] if r['arm']=='prior6' and r['seed']==seed)
            # Compare the same initial budget even if treatment subsequently extends.
            ck=io.ROOT/search['history'][0]['candidate']['learner']
            evaluations=[io.read(ck.parent/f'development_{a}.json') for a in plan['action_replicas']]
            profiles=[e['profile'] for e in evaluations]
            opponent_rates=[{r['foe']:sum(x['outcome']=='kill' for x in r['episodes'])/len(r['episodes']) for r in e['results']} for e in evaluations]
            pair.append(dict(win_rate=sum(p['mean_win_rate'] for p in profiles)/len(profiles),
                rank=search['history'][0]['candidate']['rank'],learner=io.relative(ck),
                per_opponent_win={s['id']:sum(p[s['id']] for p in opponent_rates)/len(opponent_rates) for s in plan['development_opponents']}))
        rows.append(dict(seed=seed,control=pair[0],entropy_zero=pair[1],
            win_gain=pair[1]['win_rate']-pair[0]['win_rate'],
            lower_quarter_gain=pair[1]['rank'][1]-pair[0]['rank'][1],
            worst_score_gain=pair[1]['rank'][2]-pair[0]['rank'][2],
            per_opponent_gain={k:pair[1]['per_opponent_win'][k]-pair[0]['per_opponent_win'][k] for k in pair[0]['per_opponent_win']}))
    result=dict(status='matched_entropy_comparison_verified',matched_learners=len(rows),
        first_chunk_steps_per_learner=plan['chunk_steps'],rows=rows,
        mean_win_gain=sum(r['win_gain'] for r in rows)/len(rows),
        treatment_mean_win=sum(r['entropy_zero']['win_rate'] for r in rows)/len(rows),
        control_mean_win=sum(r['control']['win_rate'] for r in rows)/len(rows),
        own_initial_win=raw['searches'][0]['initial_sampled_win_rate'],
        pure_teacher_win=raw['searches'][0]['pure_teacher_win_rate'],
        treatment_total_steps=raw['training_steps'],policy_promoted=False,
        scope='Two matched independently initialized learners per treatment, two action RNGs per policy. Reused development conditions, not heldout. First equal-budget checkpoints compared regardless of any subsequent extension. No generalization or statistical-significance claim from two repeats.',
        input_sha256={io.relative(p):io.sha(p) for p in [Path(__file__).resolve(),source/'raw_extend_analysis.json',folder/'raw_extend_analysis.json']})
    io.write(folder/'matched_entropy_analysis.json',result)
    return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    r=analyze(a.folder.resolve())
    print(r['status'],{k:r[k] for k in ['treatment_mean_win','control_mean_win','mean_win_gain','own_initial_win']})
