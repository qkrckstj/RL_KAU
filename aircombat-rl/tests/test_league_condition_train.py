from copy import deepcopy
import pytest
import tools.league_condition_train as module
from tests.test_league_interception_gate import configuration
import tests.test_league_regret_train as prior_tests


def test_archive_keeps_existing_entries_but_skips_new_exact_hybrid_copy(tmp_path):
    config = configuration()
    a = module.export(tmp_path/'a', config, 'a')
    existing_copy = module.export(tmp_path/'old_copy', config, 'old_copy')
    new_copy = module.export(tmp_path/'new_copy', config, 'new_copy')
    different = deepcopy(config); different['parent'][3] += .1
    b = module.export(tmp_path/'b', different, 'b')
    result, aliases = module.append_distinct_hybrids([a, existing_copy], [new_copy, b], a)
    assert result == [a, existing_copy, b]
    assert aliases == {'new_copy': 'a'}


def test_equal_parameters_with_different_code_are_not_merged(tmp_path):
    config = configuration()
    a = module.export(tmp_path/'a', config, 'a')
    b = module.export(tmp_path/'b', config, 'b')
    path = tmp_path/'b/policy.py'
    path.write_text(path.read_text(encoding='utf-8')+'\n# distinct source revision\n', encoding='utf-8')
    hashes = module.read(tmp_path/'b/artifact_sha256.json'); hashes['policy.py'] = module.sha(path)
    module.write(tmp_path/'b/artifact_sha256.json', hashes)
    result, aliases = module.append_distinct_hybrids([a], [b], a)
    assert result == [a, b] and aliases == {}


def test_changed_sidecar_cannot_be_used_for_deduplication(tmp_path):
    config = configuration()
    a = module.export(tmp_path/'a', config, 'a')
    b = module.export(tmp_path/'b', config, 'b')
    changed = deepcopy(config); changed['parent'][3] += .1
    module.write(tmp_path/'b/policy_net.json', changed)
    with pytest.raises(ValueError, match='JSON and loaded weights differ'):
        module.append_distinct_hybrids([a], [b], a)


def test_unchanged_searches_skip_selection_matches_and_final(tmp_path, monkeypatch):
    anchor, parent = dict(id='anchor'), dict(id='parent')
    plan = dict(anchor=anchor, parent=parent, workers=1, seeds=[1,2], opponents=[dict(id='foe')],
                development_band=100, development_n=20)
    def freeze(out, previous, damage, workers=1):
        module.write(out/'plan.json', plan)
        return plan
    monkeypatch.setattr(module, 'freeze', freeze)
    monkeypatch.setattr(module, 'verify', lambda plan: None)
    monkeypatch.setattr(module, 'search', lambda *args: dict(selected=dict(spec=anchor)))
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    calls=[]
    class Evaluate:
        def __call__(self, *args):
            calls.append(args)
            return [dict(request=dict(spec=anchor)), dict(request=dict(spec=parent))]
    monkeypatch.setattr(module, 'Evaluation', Evaluate)
    out=tmp_path/'out'; module.run(out, tmp_path/'prior', tmp_path/'damage', workers=1)
    result=module.read(out/'completion.json')
    assert len(calls)==1
    assert result['selection_games_executed']==0 and not result['final_opened']
    assert module.read(out/'frozen_selection.json')['chosen']==anchor


def test_regression_ranking_contract_is_retained(monkeypatch):
    monkeypatch.setattr(prior_tests, 'module', module)
    prior_tests.test_higher_average_with_catastrophic_regression_is_penalized()
    prior_tests.test_broad_improvement_has_no_regression_cost_and_can_be_retained()


def test_paired_reference_and_guarded_search_contract_is_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(prior_tests, 'module', module)
    prior_tests.test_search_keeps_paired_anchor_in_confirmation_and_rejects_average_trap(tmp_path, monkeypatch)


def test_new_candidates_still_use_frozen_selection_and_both_final_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(prior_tests, 'module', module)
    prior_tests.test_final_contract_still_freezes_choice_and_checks_both_baselines(tmp_path, monkeypatch)
