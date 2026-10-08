import pytest
from tools import league_temporal_audit_v2 as audit


def test_transient_windows_reader_lock_is_retried(monkeypatch):
    calls=[]; sleeps=[]
    def flaky(path,value):
        calls.append((path,value))
        if len(calls)<3:
            error=PermissionError('reader lock'); error.winerror=5
            raise error
    monkeypatch.setattr(audit,'atomic_write',flaky)
    monkeypatch.setattr(audit.time,'sleep',sleeps.append)
    audit.write('progress',{'stage':'waiting'})
    assert len(calls)==3 and len(sleeps)==2
    assert all(call==calls[0] for call in calls)


def test_persistent_permission_failure_is_not_silenced(monkeypatch):
    calls=[]
    def locked(*args):
        calls.append(1); error=PermissionError('locked'); error.winerror=32
        raise error
    monkeypatch.setattr(audit,'atomic_write',locked)
    monkeypatch.setattr(audit.time,'sleep',lambda s:None)
    with pytest.raises(PermissionError): audit.write('progress',{})
    assert len(calls)==10


def test_queue_writes_waiting_status_once_and_skips_failed_search(tmp_path,monkeypatch):
    empty=dict(source_sha256={},input_sha256={})
    search=tmp_path/'search'; search.mkdir(); panel=tmp_path/'panel.json'; out=tmp_path/'audit'
    audit.write(search/'plan.json',empty); audit.write(panel,empty)
    audit.write(search/'completion.json',dict(status='evaluation_failed'))
    monkeypatch.setattr(audit,'relative',str)
    monkeypatch.setattr(audit,'verify',lambda p:None)
    monkeypatch.setattr(audit,'sha',lambda p:'frozen')
    monkeypatch.setattr(audit,'environment',lambda:{})
    live=iter([True,True,False])
    monkeypatch.setattr(audit,'process_live',lambda pid:next(live))
    monkeypatch.setattr(audit.time,'sleep',lambda s:None)
    monkeypatch.setattr(audit,'run',lambda *a:pytest.fail('Audit must not run after failed final'))
    original=audit.atomic_write; written=[]
    def record(path,value):
        written.append((path,value)); return original(path,value)
    monkeypatch.setattr(audit,'atomic_write',record)
    result=audit.queued(out,search,panel,16,123)
    assert result['panel_opened'] is False
    assert sum(v.get('stage')=='waiting_for_search' for _,v in written)==1
