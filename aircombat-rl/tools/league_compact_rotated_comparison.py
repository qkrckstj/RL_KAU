"""Compare completed compact and rotated pilots at matched RNG and budgets."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_policy_assessment_metrics import gains
from tools.league_tournament_metrics import code_groups, group_weights


def analyze(root):
    p=io.read(root/'plan.json');io.verify(p)
    control=io.ROOT/p['paired_roster_control'];q=io.read(control/'plan.json');io.verify(q)
    done=io.read(root/'completion.json');prior=io.read(control/'completion.json')
    paths=[Path(__file__).resolve()]
    for run in (root,control):
        raw=io.read(run/'raw_compact_analysis.json')
        assert raw['status']=='compact_ppo_raw_verified' and raw['additional_steps']==2621440
        for name,digest in raw['input_sha256'].items():assert io.sha(io.ROOT/name)==digest
        paths += [run/'plan.json',run/'completion.json',run/'raw_compact_analysis.json']
    for key in ['seeds','training_bands','development_band','development_n','development_opponents',
                'warm_start','source_checkpoint_steps','source_ppo_configuration','teacher',
                'chunk_steps','base_chunks','maximum_chunks','processes','workers','action_replicas']:
        assert p[key]==q[key],key
    assert len(p['opponents'])==len(q['opponents'])==48
    assert len(set(p['groups'])&set(q['groups']))==40
    assert set(p['groups'].values())==set(q['groups'].values())
    def read_record(f):paths.append(f);return io.read(f)
    baseline=[read_record(root/'rotated'/f'baseline_action_{a}.json') for a in p['action_replicas']]
    old_base=[read_record(control/'compact'/f'baseline_action_{a}.json') for a in p['action_replicas']]
    for current,old in zip(baseline,old_base):
        assert current['request']==old['request'] and current['profile']==old['profile']
        for a,b in zip(current['results'],old['results']):
            assert {k:v for k,v in a.items() if k!='elapsed_seconds'}=={k:v for k,v in b.items() if k!='elapsed_seconds'}
    mean_win=lambda ds:sum(d['profile']['mean_win_rate'] for d in ds)/len(ds)
    groups=code_groups(p['development_opponents'])
    weights={'uniform':{k:1/len(groups) for k in groups},'code_group':group_weights(groups)}
    rows=[]
    for search in done['searches']:
        before=next(s for s in prior['searches'] if s['seed']==search['seed'])
        assert search['initial_parameters_sha256']==before['initial_parameters_sha256']
        assert search['additional_steps']==before['additional_steps']==1310720
        current=search['history'][-1]['candidate'];old=before['history'][-1]['candidate']
        def records(c):return [read_record((io.ROOT/c['learner']).parent/f'development_{a}.json') for a in p['action_replicas']]
        new_records,old_records=records(current),records(old)
        comparisons={}
        for label,records_list in [('original_compact',old_records),('baseline',baseline)]:
            comparisons[label]={key:gains([d['results'] for d in new_records],[d['results'] for d in records_list],
                p['development_opponents'],p['development_band'],p['development_n'],w) for key,w in weights.items()}
        matchups={}
        for foe in p['development_opponents']:
            key=foe['id'];rates=[]
            for ds in (baseline,old_records,new_records):
                rates.append(sum(next(r['summary']['rate'] for r in d['results'] if r['foe']==key) for d in ds)/2)
            matchups[key]=dict(baseline=rates[0],original=rates[1],rotated=rates[2],new_minus_baseline=rates[2]-rates[0],new_minus_original=rates[2]-rates[1])
        rank=current['rank'];base=done['baseline']['rank']
        passing=(rank[0]>=base[0]+.005-1e-12 and rank[1]>=base[1]-1e-12 and rank[2]>=base[2]-1e-12
                 and mean_win(new_records)>=mean_win(baseline)-1e-12)
        rows.append(dict(seed=search['seed'],baseline_win=mean_win(baseline),original_win=mean_win(old_records),
            rotated_win=mean_win(new_records),original_rank=old['rank'],rotated_rank=rank,
            retained_step=search['selected']['step'],development_extension_screen_passed=passing,
            comparisons=comparisons,matchups=matchups))
    result=dict(status='matched_roster_comparison_verified',searches=rows,
        mean_final_rotated_win=sum(s['rotated_win'] for s in rows)/len(rows),
        mean_final_original_win=sum(s['original_win'] for s in rows)/len(rows),
        passing_repeats=sum(s['development_extension_screen_passed'] for s in rows),
        scope='Same learned ancestor,RNGs,IC bands,PPO settings,teacher,reward,40-foe development and budget; change8active opponents and consequent sampling probabilities. Small consumed development,not final/unseen evidence. Initial tensors and all baseline episode records match exactly. No promotion.',
        input_sha256={io.relative(f):io.sha(f) for f in paths})
    io.write(root/'matched_roster_comparison.json',result);return result


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('folder',type=Path);a=parser.parse_args()
    r=analyze(a.folder.resolve());print(r['status'],r['mean_final_original_win'],r['mean_final_rotated_win'],r['passing_repeats'])
