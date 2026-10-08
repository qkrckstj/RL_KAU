"""Raw checkpoints, actual curriculum, settings, retention and extension audit."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from tools.league_history_clone_matched_check import ppo_configuration
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_sampled_ppo_continue import pair_rank
from tools.league_tournament_metrics import profile,code_groups


def analyze(out):
    p=io.read(out/'plan.json');io.verify(p);done=io.read(out/'completion.json')
    assert done['status']==('prioritized_smoke_complete' if p['smoke'] else 'prioritized_pilots_complete')
    paths=[out/'plan.json',out/'completion.json',Path(__file__).resolve()];games=0
    for f in list(out.glob('*/baseline_action_*.json'))+list(out.glob('*/s*/checkpoints/step_*/development_*.json')):
        d=io.read(f);paths.append(f);q=d['request']
        assert q['opponents']==p['development_opponents'] and q['band']==p['development_band'] and q['n']==p['development_n']
        validate_records(d['results'],q['opponents'],q['band'],q['n']);assert all(r['own']==q['spec']['id'] for r in d['results'])
        assert profile(d['results'],code_groups(q['opponents']))==d['profile']
        games+=d['executed_new_games'];assert d['reused_games']==0
    ps=[io.read(out/p['arms'][0]['name']/f'baseline_action_{a}.json')['profile'] for a in p['action_replicas']]
    initial=pair_rank(ps);assert initial==done['baseline']['rank']
    assert {(r['arm'],r['seed']) for r in done['searches']}=={(a['name'],s) for a in p['arms'] for s in p['seeds']}
    rows=[]
    for r in done['searches']:
        arm=next(a for a in p['arms'] if a['name']==r['arm']);root=out/r['arm']/f's{r["seed"]}'
        f=root/'loaded_ppo_configuration.json';cfg=io.read(f);paths.append(f)
        assert cfg['before']==cfg['actual']==p['source_ppo_configuration'] and cfg['buffer_gamma']==cfg['actual']['gamma'] and cfg['buffer_gae_lambda']==cfg['actual']['gae_lambda']
        assert cfg['n_steps']==cfg['buffer_size']==640 and cfg['n_envs']==32
        f=root/'actual_curriculum.json';curr=io.read(f);paths.append(f);assert curr['arm']==r['arm'] and len(curr['workers'])==32
        for s in curr['workers']:
            assert s['period']==arm['coverage_period'] and s['reward_gamma']==cfg['actual']['gamma']
            assert np.allclose(s['probabilities'],arm['probabilities'],rtol=0,atol=1e-15)
        f=root/'learning_rate_configuration.json';rate_record=io.read(f);paths.append(f)
        rate=arm['learning_rate'];wanted=dict(p['source_ppo_configuration'],learning_rate_samples=[rate]*3)
        assert rate_record['before']==p['source_ppo_configuration'] and rate_record['actual']==wanted
        assert rate_record['pretraining_optimizer_rates']==[p['source_ppo_configuration']['learning_rate_samples'][0]]
        best=done['baseline'];history=[]
        for i,h in enumerate(r['history'],1):
            c=h['candidate'];assert c['step']==h['steps']==i*p['chunk_steps'];ck=io.ROOT/c['learner'];paths.append(ck)
            ps=[io.read(ck.parent/f'development_{a}.json')['profile'] for a in p['action_replicas']];assert pair_rank(ps)==c['rank']
            saved=PPO.load(ck,device='cpu')
            assert ppo_configuration(saved)==wanted and all(g['lr']==rate for g in saved.policy.optimizer.param_groups)
            del saved
            if tuple(c['rank'])>tuple(best['rank']):best=c
            assert h['retained']==best
            history.append(dict(steps=c['step'],win_rate=sum(x['mean_win_rate'] for x in ps)/len(ps),rank=c['rank'],retained_step=best['step']))
            for a in c['replicas']:
                s=a['spec'];paths+=list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
        assert r['selected']==best and r['additional_steps']==len(history)*p['chunk_steps']
        if p['smoke']:assert r['learner_optimizer_restore_exact'] and r['initial_parameters_sha256']!=r['final_parameters_sha256'] and len(history)==1 and not r['extended']
        else:
            assert not r['extended'] and len(history)==1
        rows.append(dict(seed=r['seed'],arm=r['arm'],history=history,selected=best,extended=r['extended']))
    assert len({r['initial_parameters_sha256'] for r in done['searches']})==1
    if not p['smoke'] and p.get('matched_control_run'):
        controls=io.read(io.ROOT/p['matched_control_run']/'completion.json')['searches']
        for r in done['searches']:
            control=next(c for c in controls if c['seed']==r['seed'])
            assert r['initial_parameters_sha256']==control['initial_parameters_sha256']
    assert sum(r['additional_steps'] for r in done['searches'])==done['additional_steps']
    result=dict(status='compact_ppo_raw_verified',additional_steps=done['additional_steps'],evaluation_games=games,initial_rank=initial,searches=rows,policy_promoted=False,scope=p['quality_scope'],input_sha256={io.relative(f):io.sha(f) for f in paths})
    io.write(out/'raw_compact_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['additional_steps'])

