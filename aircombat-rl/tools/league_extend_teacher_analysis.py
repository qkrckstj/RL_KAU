"""Verify fresh initializations, teacher guard and sampled residual PPO results."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import code_groups,profile


def analyze(folder):
    plan=io.read(folder/'plan.json');io.verify(plan);done=io.read(folder/'completion.json')
    assert done['status']==('extend_teacher_smoke_complete' if plan['smoke'] else 'extend_teacher_pilots_complete')
    assert {(r['arm'],r['seed']) for r in done['searches']}=={(a['name'],s) for a in plan['arms'] for s in plan['seeds']}
    paths=[folder/'plan.json',folder/'completion.json',Path(__file__).resolve()];games=0
    files=[folder/'teacher_development.json']+list(folder.glob('*/baseline_action_*.json'))+list(folder.glob('*/s*/checkpoints/step_*/development_*.json'))
    for path in files:
        row=io.read(path);paths.append(path);req=row['request']
        assert req['opponents']==plan['development_opponents'] and req['band']==plan['development_band'] and req['n']==plan['development_n']
        validate_records(row['results'],req['opponents'],req['band'],req['n']);assert all(r['own']==req['spec']['id'] for r in row['results'])
        assert profile(row['results'],code_groups(req['opponents']))==row['profile']
        assert row['reused_games']==0 and row['executed_new_games']==len(req['opponents'])*req['n'];games+=row['executed_new_games']
    teacher=io.read(folder/'teacher_development.json')['profile'];reference=pair_rank([teacher,teacher]);assert reference==done['teacher_reference_rank']
    baselines={}
    for arm in plan['arms']:
        ps=[io.read(folder/arm['name']/f'baseline_action_{a}.json')['profile'] for a in plan['action_replicas']]
        assert pair_rank(ps)==done['baselines'][arm['name']]['rank'];baselines[arm['name']]=ps
    summaries=[]
    for r in done['searches']:
        root=folder/r['arm']/f's{r["seed"]}';init=plan['initializations'][r['arm']][str(r['seed'])];assert r['initial_parameters_sha256']==init['parameter_digest'] and init['source_steps']==0 and init['optimizer_has_no_history'] and init['initial_logits_exact']
        path=root/'loaded_ppo_configuration.json';loaded=io.read(path);paths.append(path)
        assert loaded['actual']==plan['source_ppo_configuration'] and loaded['n_steps']==640 and loaded['prior_bias']==r['prior_bias'] and loaded['source_checkpoint_steps']==0 and loaded['initial_residual_zero']
        best=done['baselines'][r['arm']]
        for i,h in enumerate(r['history'],1):
            c=h['candidate'];assert c['step']==h['steps']==i*plan['chunk_steps'];ck=io.ROOT/c['learner'];paths.append(ck)
            ps=[io.read(ck.parent/f'development_{a}.json')['profile'] for a in plan['action_replicas']];assert pair_rank(ps)==c['rank']
            if tuple(c['rank'])>tuple(best['rank']):best=c
            assert h['retained']==best
            for a in c['replicas']:
                s=a['spec'];paths+=list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
        assert r['selected']==best and r['additional_steps']==len(r['history'])*plan['chunk_steps']
        initial=done['baselines'][r['arm']]['rank'];first=r['history'][0]['retained']['rank']
        expected=not plan['smoke'] and first[0]>=max(initial[0],reference[0])+.005 and first[1]>=max(initial[1],reference[1]) and first[2]>=max(initial[2],reference[2])
        assert r['extended']==expected and len(r['history'])==(2 if expected else 1)
        if plan['smoke']:assert r['learner_optimizer_restore_exact'] and r['initial_parameters_sha256']!=r['final_parameters_sha256']
        summaries.append(dict(arm=r['arm'],seed=r['seed'],steps=r['additional_steps'],final_win_rate=sum(p['mean_win_rate'] for p in ps)/2,initial_sampled_win_rate=sum(p['mean_win_rate'] for p in baselines[r['arm']])/2,pure_teacher_win_rate=teacher['mean_win_rate'],final_rank=c['rank'],teacher_rank=reference,retained_step=best['step'],extended=r['extended']))
    if plan['smoke']:
        paths.append(folder/'qualification.json');q=io.read(folder/'qualification.json');assert q['status']=='greedy_teacher_exact'
        for p in sorted((folder/'qualification').glob('check_*.json')):
            row=io.read(p);paths.append(p);assert row['exact_episode_summary_equality'] and row['direct']['episodes']==row['wrapped']['episodes'] and row['direct']['summary']==row['wrapped']['summary']
    else:
        for a in plan['arms']:assert len({r['initial_parameters_sha256'] for r in done['searches'] if r['arm']==a['name']})==len(plan['seeds'])
    assert sum(r['additional_steps'] for r in done['searches'])==done['additional_steps']
    result=dict(status='extend_teacher_raw_verified',searches=summaries,training_steps=done['additional_steps'],evaluation_games=games,teacher_reference=teacher,input_sha256={io.relative(p):io.sha(p) for p in paths},policy_promoted=False,scope='Fresh neural seeds sharing the same frozen teacher; action replicates are not independent learning repeats. Pure teacher is a deterministic reference. Development-only result, not final/heldout evidence.')
    io.write(folder/'raw_extend_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['training_steps']);print(r['searches'])
