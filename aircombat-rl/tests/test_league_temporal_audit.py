from copy import deepcopy
from pathlib import Path
import pytest
from tools import league_temporal_audit as audit


def test_rejects_failed_test_posthoc_switch_and_unfrozen_choice():
    completed=dict(status='evaluation_passed',selected={'id':'selected'},selected_before_test=True)
    selected=dict(chosen={'id':'selected'})
    audit.validate_selection(completed,selected)
    for patch in ({'status':'evaluation_failed'},{'selected':{'id':'other'}},{'selected_before_test':False}):
        changed=deepcopy(completed); changed.update(patch)
        with pytest.raises(ValueError): audit.validate_selection(changed,selected)


def test_unsuccessful_search_never_opens_panel_or_runs_matches(tmp_path,monkeypatch):
    empty=dict(source_sha256={},input_sha256={})
    search=tmp_path/'search'; search.mkdir()
    panel=tmp_path/'panel.json'
    out=tmp_path/'audit'
    audit.write(search/'plan.json',empty); audit.write(panel,empty)
    audit.write(search/'completion.json',dict(status='no_development_improvement'))
    monkeypatch.setattr(audit,'relative',lambda p: str(p))
    monkeypatch.setattr(audit,'verify',lambda p: None)
    monkeypatch.setattr(audit,'sha',lambda p: 'frozen')
    monkeypatch.setattr(audit,'environment',lambda: {})
    monkeypatch.setattr(audit,'process_live',lambda pid: False)
    monkeypatch.setattr(audit,'run',lambda *a: pytest.fail('An unsuccessful search must not open audit matches'))
    result=audit.queued(out,search,panel,16,123)
    assert result['status']=='not_eligible' and result['panel_opened'] is False
    assert not (out/'plan.json').exists()
