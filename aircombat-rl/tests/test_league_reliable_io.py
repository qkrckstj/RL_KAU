import os
import threading
import pytest
from tools.autolab_cem import read
from tools import league_reliable_io as io


@pytest.mark.skipif(os.name!='nt',reason='Windows file replacement lock')
def test_real_reader_lock_released_during_retry(tmp_path):
    path=tmp_path/'progress.json';io.write(path,{'old':True})
    reader=path.open('rb')
    closer=threading.Timer(.10,reader.close);closer.start()
    try: io.write(path,{'new':True})
    finally: closer.join();reader.close()
    assert read(path)=={'new':True}


def test_persistent_error_is_bounded(monkeypatch):
    attempts=[]
    def denied(*args):
        attempts.append(1);error=PermissionError('denied');error.winerror=5
        raise error
    monkeypatch.setattr(io,'atomic_write',denied);monkeypatch.setattr(io.time,'sleep',lambda s:None)
    with pytest.raises(PermissionError):io.write('unused',{})
    assert len(attempts)==10
