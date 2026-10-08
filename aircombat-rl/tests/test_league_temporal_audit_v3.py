import pytest
from tools import league_temporal_audit_v3 as audit
from tools.league_reliable_evaluation import evaluate_candidates
from tools.league_reliable_io import write


def test_selection_remains_frozen(monkeypatch):
    from tests import test_league_temporal_audit as prior
    monkeypatch.setattr(prior, 'audit', audit)
    prior.test_rejects_failed_test_posthoc_switch_and_unfrozen_choice()


@pytest.mark.parametrize('status', ['evaluation_failed', None])
def test_failed_or_missing_completion_never_opens_panel(tmp_path, monkeypatch, status):
    search, panel, out = tmp_path/'search', tmp_path/'panel.json', tmp_path/'audit'
    empty = dict(source_sha256={}, input_sha256={})
    write(search/'plan.json', empty); write(panel, empty)
    if status is not None:
        write(search/'completion.json', dict(status=status))
    monkeypatch.setattr(audit, 'relative', str)
    monkeypatch.setattr(audit, 'verify', lambda plan: None)
    monkeypatch.setattr(audit, 'sha', lambda path: 'frozen')
    monkeypatch.setattr(audit, 'environment', lambda: {})
    live = iter([True, True, False])
    monkeypatch.setattr(audit, 'process_live', lambda pid: next(live))
    monkeypatch.setattr(audit.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(audit, 'run', lambda *args: pytest.fail('Panel must remain unopened'))
    writes = []
    def recording_write(path, value):
        writes.append(value)
        return write(path, value)
    monkeypatch.setattr(audit, 'write', recording_write)
    result = audit.queued(out, search, panel, 16, 123)
    assert not result['panel_opened']
    assert result['status'] == ('not_eligible' if status else 'search_incomplete')
    assert sum(value.get('stage') == 'waiting_for_search' for value in writes) == 1
    assert not (out/'plan.json').exists()


def test_match_evaluator_uses_reliable_writes():
    assert audit.evaluate_candidates is evaluate_candidates
    assert audit.write is write
