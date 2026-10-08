"""Verify physical budgets, PPO settings and paired outcomes of held-action pilots."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import code_groups,profile


def analyze(folder):
    plan=io.read(folder/'plan.json');io.verify(plan);done=io.read(folder/'completion.json')
    assert done['status']==('hold_smoke_complete' if plan['smoke'] else 'hold_pilots_complete')
    assert {(r['arm'],r['seed']) for r in done['searches']}=={(a['name'],s) for a in plan['arms'] for s in plan['seeds']}
    paths=[folder/'plan.json',folder/'completion.json',Path(__file__).resolve()];games=0;records={}
    for path in list(folder.glob('*/baseline_action_*.json'))+list(folder.glob('*/s*/checkpoints/step_*/development_*.json')):
        row=io.read(path);paths.append(path);request=row['request']
        assert request['opponents']==plan['development_opponents'] and request['band']==plan['development_band'] and request['n']==plan['development_n']
        validate_records(row['results'],request['opponents'],request['band'],request['n'])
        assert all(r['own']==request['spec']['id'] for r in row['results'])
        assert profile(row['results'],code_groups(request['opponents']))==row['profile']
        assert row['reused_games']==0 and row['executed_new_games']==len(request['opponents'])*request['n']
        games+=row['executed_new_games'];records[io.relative(path)]=row
    baseline_profiles={}
    for arm in plan['arms']:
        name=arm['name'];profiles=[io.read(folder/name/f'baseline_action_{seed}.json')['profile'] for seed in plan['action_replicas']]
        assert pair_rank(profiles)==done['baselines'][name]['rank'];baseline_profiles[name]=profiles
    summaries=[]
    for r in done['searches']:
        arm=next(a for a in plan['arms'] if a['name']==r['arm']);root=folder/r['arm']/f's{r["seed"]}';path=root/'loaded_ppo_configuration.json';loaded=io.read(path);paths.append(path)
        wanted=dict(plan['source_ppo_configuration'],gamma=plan['source_ppo_configuration']['gamma']**arm['hold_steps'],gae_lambda=plan['source_ppo_configuration']['gae_lambda']**arm['hold_steps'])
        assert loaded['before']==plan['source_ppo_configuration'] and loaded['actual']==wanted
        assert loaded['n_steps']==loaded['buffer_size']==arm['n_steps'] and loaded['buffer_gamma']==wanted['gamma'] and loaded['buffer_gae_lambda']==wanted['gae_lambda']
        path=root/'physical_totals.json';counts=io.read(path);paths.append(path)
        assert counts['decision_steps']==r['additional_steps']==arm['chunk_steps'] and counts['physical_steps']==r['physical_steps']
        assert r['additional_steps']<=r['physical_steps']<=r['additional_steps']*arm['hold_steps']
        if plan['smoke']:assert r['learner_optimizer_restore_exact'] and r['initial_parameters_sha256']!=r['final_parameters_sha256']
        candidate=r['history'][-1]['candidate'];ck=io.ROOT/candidate['learner'];paths.append(ck)
        profiles=[io.read(ck.parent/f'development_{seed}.json')['profile'] for seed in plan['action_replicas']]
        assert pair_rank(profiles)==candidate['rank']
        previous=done['baselines'][r['arm']];expected=candidate if tuple(candidate['rank'])>tuple(previous['rank']) else previous
        assert r['selected']==expected
        for replica in candidate['replicas']:
            spec=replica['spec'];paths += list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]
        mean=lambda ps:sum(p['mean_win_rate'] for p in ps)/2
        summaries.append(dict(arm=r['arm'],seed=r['seed'],physical_steps=r['physical_steps'],decisions=r['additional_steps'],final_win_rate=mean(profiles),initial_execution_win_rate=mean(baseline_profiles[r['arm']]),common_original_win_rate=mean(baseline_profiles['control']),final_rank=candidate['rank'],initial_execution_rank=previous['rank'],common_original_rank=done['baselines']['control']['rank'],retained_step=r['selected']['step']))
    assert sum(r['physical_steps'] for r in done['searches'])==done['total_physical_steps']
    assert sum(r['additional_steps'] for r in done['searches'])==done['total_learner_decisions']
    expected_records=2*len(plan['arms'])+2*len(done['searches']);assert len(records)==expected_records
    result=dict(status='hold_raw_verified',searches=summaries,evaluation_games=games,total_physical_steps=done['total_physical_steps'],input_sha256={io.relative(p):io.sha(p) for p in paths},scope='Development-only continuation after common pretrained prefix. Decision count differs; actual physical budget reported. Hold variant starts with different behavior. Greedy unheld auxiliary export is not the evaluated held policy.',policy_promoted=False)
    io.write(folder/'raw_hold_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['total_physical_steps']);print(r['searches'])
