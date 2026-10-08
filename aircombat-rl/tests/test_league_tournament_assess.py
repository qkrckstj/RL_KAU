import pytest
from tools import league_tournament_assess as module
from tests.test_league_regret_train import record, FOES


@pytest.mark.parametrize('final_passes', [True, False])
def test_final_uses_only_frozen_choice_and_audit_cannot_select(tmp_path, monkeypatch, final_passes):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'verify', lambda plan: None)
    monkeypatch.setattr(module, 'process_live', lambda pid: False)
    monkeypatch.setattr(module, 'resources', lambda: dict(commit_headroom_gib=50))
    plan = dict(anchor=dict(id='anchor'), parent=dict(id='parent'),
        candidates=[dict(id='c0'), dict(id='c1')], groups={f:f for f in FOES},
        opponents=[dict(id=f) for f in FOES], workers=8, profile_name='test',
        selection_band=1000, selection_n=20, final_band=2000, final_n=20)
    benchmark = tmp_path/'benchmark'
    module.write(benchmark/'completion.json', dict(median_initialized_worker_private_gib={'single_w8':.3}))
    def freeze(out, *args):
        module.write(out/'plan.json', plan)
        return plan
    monkeypatch.setattr(module, 'freeze', freeze)
    class Pool:
        def __init__(self, **kw): assert kw['max_workers']==8
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    calls=[]
    class Evaluation:
        def __call__(self, pool, specs, foes, band, n, out, traced=False):
            calls.append(([s['id'] for s in specs],band,traced))
            if traced:
                assert module.read(out.parent/'frozen_selection.json')['chosen']['id']=='c0'
            answer=[]
            for spec in specs:
                rate={'anchor':.3,'parent':.4,'c0':.9,'c1':.7}[spec['id']]
                if traced and spec['id']=='c0' and not final_passes: rate=0.
                r=record([rate]*len(FOES),spec['id'],n,band)
                for row in r['results']:
                    for e in row['episodes']: e['won']=e['outcome']=='kill'
                    if traced:
                        row['damage_traces']=[dict(seed=e['seed'],seat=e['seat'],first_hit=10.,damage_seconds=.3,
                            timeline_1s=[dict(t=0.,hp=1.)],terminal=dict(t=40.,hp=e['own_health']),
                            snapshots={'30':dict(t=30.,hp=e['own_health'])}) for e in row['episodes']]
                answer.append(r)
            return answer
    monkeypatch.setattr(module,'Evaluation',Evaluation)
    audits=[]
    def audit(pool, evaluate, p, chosen, out):
        assert module.read(out/'final_decision.json')['status']=='tournament_profile_passed'
        assert module.read(out/'frozen_selection.json')['chosen']==chosen
        audits.append(chosen['id'])
        return dict(status='audit_complete',model_selection_performed=False)
    monkeypatch.setattr(module,'temporal_audit',audit)
    out=tmp_path/'assessment'
    result=module.run(out,tmp_path/'prior',tmp_path/'audit',benchmark,tmp_path/'panel')
    assert calls==[(['anchor','parent','c0','c1'],1000,False),(['anchor','parent','c0'],2000,True)]
    assert result['selected']['id']=='c0' and result['selected_before_test']
    assert (result['status']=='tournament_profile_passed') is final_passes
    assert audits==(['c0'] if final_passes else [])
    assert (out/'final/damage_summary.json').exists()
