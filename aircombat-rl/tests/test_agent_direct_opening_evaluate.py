from experiments.direct_pilot import evaluate as original
from test_range_evaluate import RangeMatch

def test_opening_observer_preserves_all_original_decisions_and_metrics(monkeypatch):
    from experiments.direct_pilot import opening_evaluate as candidate
    created=[]
    def factory(*a):
        env=RangeMatch(*a);created.append(env);return env
    for module in (original,candidate):
        monkeypatch.setattr(module,'MatchEnv',factory)
        monkeypatch.setattr(module,'geometry',lambda obs:dict(tail_attack=False,head_on=True))
    old=original.cell({},dict(id='foe'),42,'red')
    new=candidate.cell(dict(diagnostic_opening_seconds=.1),dict(id='foe'),42,'red')
    assert all(new[k]==v for k,v in old.items() if k!='wall_seconds')
    assert created[0].actions==created[1].actions and all(e.closed for e in created)
    assert new['opening_end_public_probe'] is not None
    assert new['opening_end_public_probe']['t']>=.1
