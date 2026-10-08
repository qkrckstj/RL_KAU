"""Summarize completed compact pilots before allocating a new learning budget."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io


def analyze(root):
    plan=io.read(root/'plan.json');io.verify(plan)
    done=io.read(root/'completion.json');raw=io.read(root/'raw_compact_analysis.json')
    protocol_path=io.ROOT/'experiments/league/compact_decision_protocol_20261008.json'
    protocol=io.read(protocol_path)
    registration=io.read(root/'decision_protocol_registration.json')
    assert registration['sha256']==io.sha(protocol_path) and registration['completed_chunks']==[]
    assert not plan['smoke'] and raw['status']=='compact_ppo_raw_verified'
    assert done['additional_steps']==2621440 and len(done['searches'])==2
    assert protocol['training_run']==io.relative(root)
    for name,digest in raw['input_sha256'].items():assert io.sha(io.ROOT/name)==digest
    base=done['baseline']['rank']; rows=[]
    baseline=[io.read(root/'compact'/f'baseline_action_{a}.json') for a in plan['action_replicas']]
    mean_win=lambda ds:sum(d['profile']['mean_win_rate'] for d in ds)/len(ds)
    def rates(ds):
        return {s['id']:sum(next(r['summary']['rate'] for r in d['results'] if r['foe']==s['id']) for d in ds)/len(ds) for s in plan['development_opponents']}
    baseline_rates=rates(baseline)
    for search in done['searches']:
        c=search['history'][-1]['candidate']; rank=c['rank'];folder=(io.ROOT/c['learner']).parent
        current=[io.read(folder/f'development_{a}.json') for a in plan['action_replicas']]
        delta={k:v-baseline_rates[k] for k,v in rates(current).items()}
        ready=(rank[0]-base[0]>=protocol['minimum_combined_gain']-1e-12
            and rank[1]>=base[1]-1e-12 and rank[2]>=base[2]-1e-12
            and mean_win(current)>=mean_win(baseline)-1e-12)
        rows.append(dict(seed=search['seed'],rank=rank,combined_gain=rank[0]-base[0],
            mean_win_rate=mean_win(current),win_gain=mean_win(current)-mean_win(baseline),
            lower_quarter_gain=rank[1]-base[1],worst_gain=rank[2]-base[2],
            retained_step=search['selected']['step'],extension_screen_passed=ready,
            per_opponent_win_gain=delta,
            active_opponent_mean_gain=sum(delta[k] for k in delta if k in plan['groups'])/sum(k in plan['groups'] for k in delta),
            archived_inactive_mean_gain=sum(delta[k] for k in delta if k not in plan['groups'])/sum(k not in plan['groups'] for k in delta)))
    passing=sum(r['extension_screen_passed'] for r in rows)
    next_action=('confirm_on_small_off_roster_panel_then_consider_matched_extension' if passing else
        'do_not_expand_budget_analyze_matchup_changes_before_rotation_or_strategy_change')
    result=dict(status='compact_pilot_decision_verified',baseline_rank=base,baseline_mean_win_rate=mean_win(baseline),
        searches=rows,passing_repeats=passing,next_action=next_action,
        scope='Small consumed development screen; passing is a budget-allocation signal,not proof of general improvement. Both learners share an ancestor. No promotion.',
        input_sha256={io.relative(f):io.sha(f) for f in [root/'plan.json',root/'completion.json',root/'raw_compact_analysis.json',root/'decision_protocol_registration.json',protocol_path,Path(__file__).resolve()]})
    io.write(root/'pilot_decision.json',result);return result


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    print(analyze(args.folder.resolve()))
