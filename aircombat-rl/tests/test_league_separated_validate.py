import pytest
from tools import league_separated_validate as module
from tests.test_league_regret_train import record, FOES


@pytest.mark.parametrize('passes', [True, False])
def test_choice_is_fixed_and_final_failure_does_not_open_temporal(tmp_path,monkeypatch,passes):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'verify',lambda p:None)
    monkeypatch.setattr(module,'process_live',lambda pid:False)
    monkeypatch.setattr(module,'resources',lambda:dict(commit_headroom_gib=50))
    models=[dict(id=n) for n in ('anchor','parent','warm','chosen')]
    plan=dict(chosen=models[-1],final_models=models,opponents=[dict(id=f) for f in FOES],
        groups={f:f for f in FOES},final_band=2000,final_n=20,profile_name='test',scope='test')
    def freeze(out,*args):
        module.write(out/'plan.json',plan);module.write(out/'frozen_selection.json',dict(chosen=models[-1]))
        return plan
    monkeypatch.setattr(module,'freeze',freeze)
    class Pool:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
    monkeypatch.setattr(module,'ProcessPoolExecutor',Pool)
    calls=[]
    class Evaluation:
        def __call__(self,pool,specs,foes,band,n,out,traced=False):
            assert module.read(out.parent/'frozen_selection.json')['chosen']==models[-1]
            calls.append([s['id'] for s in specs]);assert traced
            answer=[]
            for spec in specs:
                rate={'anchor':.3,'parent':.4,'warm':.7,'chosen':.9 if passes else .2}[spec['id']]
                r=record([rate]*len(FOES),spec['id'],n,band)
                for f in r['results']:
                    for e in f['episodes']:e['won']=e['outcome']=='kill'
                    f['damage_traces']=[dict(seed=e['seed'],seat=e['seat'],first_hit=10.,damage_seconds=.3,
                        timeline_1s=[dict(t=0.,hp=1.)],terminal=dict(t=40.,hp=e['own_health']),
                        snapshots={'30':dict(t=30.,hp=e['own_health'])}) for e in f['episodes']]
                answer.append(r)
            return answer
    monkeypatch.setattr(module,'Evaluation',Evaluation)
    audits=[]
    def audit(pool,evaluate,p,chosen,out):
        assert module.read(out/'final_decision.json')['status']=='tournament_profile_passed'
        audits.append(chosen);return dict(status='audit_complete')
    monkeypatch.setattr(module,'temporal_audit',audit)
    result=module.run(tmp_path/'out',tmp_path/'previous',8)
    assert calls==[['anchor','parent','warm','chosen']]
    assert (result['status']=='tournament_profile_passed') is passes
    assert result['selected']==models[-1] and result['selected_before_test']
    assert audits==([models[-1]] if passes else [])
    assert 'warm_comparison' in result
