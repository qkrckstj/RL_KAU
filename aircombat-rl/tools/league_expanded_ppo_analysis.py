"""Verify all continuation checkpoints, retention and budget decisions."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import profile,code_groups


def analyze(out):
    plan=io.read(out/'plan.json');io.verify(plan);done=io.read(out/'completion.json')
    assert done['status']==('expanded_ppo_smoke_complete' if plan['smoke'] else 'expanded_ppo_complete')
    paths=[out/'plan.json',out/'completion.json',Path(__file__).resolve()];evaluation_games=0
    for f in list(out.glob('league/baseline_action_*.json'))+list(out.glob('league/s*/checkpoints/step_*/development_*.json')):
        d=io.read(f);paths.append(f);q=d['request']
        assert q['opponents']==plan['development_opponents'] and q['band']==plan['development_band'] and q['n']==plan['development_n']
        validate_records(d['results'],q['opponents'],q['band'],q['n']);assert all(r['own']==q['spec']['id'] for r in d['results'])
        assert profile(d['results'],code_groups(q['opponents']))==d['profile']
        evaluation_games+=d['executed_new_games'];assert d['reused_games']==0
    initial=pair_rank([io.read(out/f'league/baseline_action_{a}.json')['profile'] for a in plan['action_replicas']]);assert initial==done['baseline']['rank']
    assert {r['seed'] for r in done['searches']}==set(plan['seeds'])
    rows=[]
    for r in done['searches']:
        root=out/f'league/s{r["seed"]}';f=root/'loaded_ppo_configuration.json';cfg=io.read(f);paths.append(f)
        assert cfg['before']==cfg['actual']==plan['source_ppo_configuration'] and cfg['n_steps']==cfg['buffer_size']==640 and cfg['n_envs']==32
        assert cfg['buffer_gamma']==plan['source_ppo_configuration']['gamma'] and cfg['buffer_gae_lambda']==plan['source_ppo_configuration']['gae_lambda']
        best=done['baseline'];history=[]
        for i,h in enumerate(r['history'],1):
            c=h['candidate'];assert c['step']==h['steps']==i*plan['chunk_steps']
            ck=io.ROOT/c['learner'];paths.append(ck)
            ps=[io.read(ck.parent/f'development_{a}.json')['profile'] for a in plan['action_replicas']];assert pair_rank(ps)==c['rank']
            if tuple(c['rank'])>tuple(best['rank']):best=c
            assert h['retained']==best
            history.append(dict(steps=c['step'],win_rate=sum(p['mean_win_rate'] for p in ps)/len(ps),rank=c['rank'],retained_step=best['step']))
            for a in c['replicas']:
                s=a['spec'];paths+=list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
        assert r['selected']==best and r['additional_steps']==len(r['history'])*plan['chunk_steps']
        if plan['smoke']:
            assert r['learner_optimizer_restore_exact'] and r['initial_parameters_sha256']!=r['final_parameters_sha256'] and len(history)==1 and not r['extended']
        else:
            prev,latest=[h['retained']['rank'] for h in r['history'][:2]]
            expected=latest[0]-prev[0]>=.005-1e-12 and latest[0]-initial[0]>=.005-1e-12 and latest[1]>=max(prev[1],initial[1])-1e-12 and latest[2]>=max(prev[2],initial[2])
            assert r['extended']==expected and len(history)==(3 if expected else 2)
        rows.append(dict(seed=r['seed'],history=history,selected=best,extended=r['extended']))
    assert len({r['initial_parameters_sha256'] for r in done['searches']})==1
    assert sum(r['additional_steps'] for r in done['searches'])==done['additional_steps']
    result=dict(status='expanded_ppo_raw_verified',additional_steps=done['additional_steps'],source_prefix_steps=plan['source_checkpoint_steps'],evaluation_games=evaluation_games,searches=rows,initial_rank=initial,policy_promoted=False,scope=plan['quality_scope'],input_sha256={io.relative(p):io.sha(p) for p in paths})
    io.write(out/'raw_expanded_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['additional_steps'])
