from pathlib import Path
from tools.league_live_status import snapshot


def test_status_does_not_open_live_files(tmp_path,monkeypatch):
    generation=tmp_path/'s2801/g5'
    match=generation/'screen/fresh_example/candidate_002_matches/foe_013.json'
    match.parent.mkdir(parents=True)
    match.write_text('content is intentionally not parsed',encoding='utf-8')
    prior=tmp_path/'s2801/g4';prior.mkdir()
    (prior/'state.json').write_text('{}',encoding='utf-8')
    (generation/'screen/progress.json').write_text('in progress',encoding='utf-8')
    def forbidden(*args,**kwargs):
        raise AssertionError('Live status must use metadata rather than file contents')
    monkeypatch.setattr(Path,'open',forbidden)
    result=snapshot(tmp_path)
    assert result['searches'][0]['completed_generation_markers']==1
    assert result['searches'][0]['batches'][0]['saved_match_jobs']==1
    assert result['searches'][0]['batches'][0]['candidate_records']==0
