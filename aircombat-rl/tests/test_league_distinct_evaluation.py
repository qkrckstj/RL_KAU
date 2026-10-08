import pytest
from tools import league_distinct_evaluation as module


def test_repeated_parent_is_evaluated_once_without_losing_search_indices(tmp_path, monkeypatch):
    calls = []
    def evaluator(pool, candidates, opponents, band, n, out, weights):
        calls.append(candidates)
        return [dict(request=dict(spec=s), results=[{'wins':i}]) for i,s in enumerate(candidates)]
    monkeypatch.setattr(module, 'evaluate_candidates', evaluator)
    parent = dict(id='parent', kind='reactive', parameters=[1, 2])
    challenger = dict(id='new', kind='reactive', parameters=[3, 4])
    models = [parent, dict(parent), dict(parent), challenger]
    records = module.evaluate_distinct_candidates(None, models, [], 500, 4, tmp_path)
    assert calls == [[parent, challenger]]
    assert [r['request']['spec'] for r in records] == models
    assert records[0] == records[1] == records[2]
    records[0]['results'][0]['wins'] = 99
    assert records[1]['results'][0]['wins'] == 0
    with pytest.raises(ValueError, match='frozen record'):
        module.evaluate_distinct_candidates(None, models, [], 501, 4, tmp_path)
    assert len(calls) == 1


def test_same_id_does_not_merge_distinct_checkpoints(tmp_path, monkeypatch):
    models = [dict(id='same',kind='submission',design='model',weights=w)
              for w in ('one.zip','two.zip')]
    def evaluator(pool, candidates, *args):
        assert candidates == models
        return [dict(request=dict(spec=s)) for s in candidates]
    monkeypatch.setattr(module, 'evaluate_candidates', evaluator)
    records = module.evaluate_distinct_candidates(None, models, [], 500, 4, tmp_path)
    assert len(records) == 2
