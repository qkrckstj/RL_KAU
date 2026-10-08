"""Verify matched gamma, raw development episodes and policy retention."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import profile,code_groups


def analyze(out):
    plan=io.read(out/'plan.json');io.verify(plan);done=io.read(out/'completion.json')
    assert done['status']==('discount_smoke_complete' if plan['smoke'] else 'discount_pilots_complete')
    paths=[out/'plan.json',out/'completion.json',Path(__file__).resolve()];games=0
    for f in list(out.glob('*/baseline_action_*.json'))+list(out.glob('*/s*/checkpoints/step_*/development_*.json')):
        d=io.read(f);paths.append(f);q=d['request']
        assert q['opponents']==plan['development_opponents'] and q['band']==plan['development_band'] and q['n']==plan['development_n']
        validate_records(d['results'],q['opponents'],q['band'],q['n']);assert all(r['own']==q['spec']['id'] for r in d['results'])
        assert profile(d['results'],code_groups(q['opponents']))==d['profile']
        games+=d['executed_new_games'];assert d['reused_games']==0
    initial_profiles=[io.read(out/f'control/baseline_action_{a}.json')['profile'] for a in plan['action_replicas']]
    assert pair_rank(initial_profiles)==done['baseline']['rank']
    assert {(r['arm'],r['seed']) for r in done['searches']}=={(a['name'],s) for a in plan['arms'] for s in plan['seeds']}
    rows=[]
    for r in done['searches']:
        arm=next(a for a in plan['arms'] if a['name']==r['arm'])
        f=out/r['arm']/f's{r["seed"]}'/'loaded_ppo_configuration.json';cfg=io.read(f);paths.append(f)
        assert cfg['before']==plan['source_ppo_configuration'] and cfg['actual']==dict(cfg['before'],gamma=arm['gamma'])
        assert cfg['buffer_gamma']==arm['gamma'] and cfg['buffer_gae_lambda']==cfg['before']['gae_lambda']
        assert cfg['n_steps']==cfg['buffer_size']==640 and cfg['n_envs']==32
        assert len(cfg['reward_wrappers'])==32 and all(s==dict(reward_gamma=arm['gamma'],shaping=True) for s in cfg['reward_wrappers'])
        assert len(r['history'])==1 and not r['extended'] and r['additional_steps']==plan['chunk_steps']
        h=r['history'][0];c=h['candidate'];assert c['step']==h['steps']==plan['chunk_steps']
        ck=io.ROOT/c['learner'];paths.append(ck)
        ps=[io.read(ck.parent/f'development_{a}.json')['profile'] for a in plan['action_replicas']]
        assert pair_rank(ps)==c['rank']
        best=c if tuple(c['rank'])>tuple(done['baseline']['rank']) else done['baseline']
        assert h['retained']==r['selected']==best
        if plan['smoke']:assert r['learner_optimizer_restore_exact'] and r['initial_parameters_sha256']!=r['final_parameters_sha256']
        for a in c['replicas']:
            s=a['spec'];paths+=list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
        rows.append(dict(arm=r['arm'],seed=r['seed'],win_rate=sum(p['mean_win_rate'] for p in ps)/len(ps),rank=c['rank'],selected=best))
    assert len({r['initial_parameters_sha256'] for r in done['searches']})==1
    assert sum(r['additional_steps'] for r in done['searches'])==done['additional_steps']==len(plan['arms'])*len(plan['seeds'])*plan['chunk_steps']
    result=dict(status='discount_raw_verified',additional_steps=done['additional_steps'],evaluation_games=games,initial_win_rate=sum(p['mean_win_rate'] for p in initial_profiles)/len(initial_profiles),searches=rows,policy_promoted=False,scope=plan['quality_scope'],input_sha256={io.relative(p):io.sha(p) for p in paths})
    io.write(out/'raw_discount_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['additional_steps'])
