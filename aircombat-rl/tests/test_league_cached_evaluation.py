import json
import pytest
from tools import league_cached_evaluation as module


def setup(monkeypatch):
    calls=[]
    def evaluator(pool,candidates,opponents,band,n,out,weights):
        calls.append(candidates)
        return [dict(request=dict(spec=s,opponents=opponents,band=band,n=n,weights=weights),
            results=[dict(own=s['id'],foe=f['id'],summary=dict(n=n),episodes=[
                dict(seed=band+i,seat=seat) for i in range(n//2) for seat in ('red','blue')])
                for f in opponents],metrics={}) for s in candidates]
    monkeypatch.setattr(module,'evaluate_candidates',evaluator)
    return calls


def test_exact_reuse_across_searches_and_context_changes(tmp_path,monkeypatch):
    calls=setup(monkeypatch)
    model=dict(id='parent',kind='fixed',action=0);foes=[dict(id='ace',kind='bot',name='ace')]
    context=dict(plan_sha256='a'*64,runtime_sha256='b'*64,deterministic=True)
    def run(name,**changes):
        args=dict(pool=None,candidates=[model,dict(model)],opponents=foes,band=100,n=4,
                  out=tmp_path/name,cache=tmp_path/'cache',context=context)
        args.update(changes);return module.evaluate_cached(**args)
    first=run('search0');second=run('search1')
    assert len(calls)==1 and len(calls[0])==1 and first==second
    second[0]['results'][0]['episodes'][0]['seed']=-1
    assert second[1]['results'][0]['episodes'][0]['seed']==100
    run('new_band',band=200)
    run('new_runtime',context={**context,'runtime_sha256':'c'*64})
    run('new_policy',candidates=[{**model,'action':1}])
    run('new_opponent',opponents=[dict(id='lead',kind='bot',name='lead')])
    assert len(calls)==5
    with pytest.raises(ValueError,match='frozen record'):run('search0',band=300)


def test_corrupt_cache_is_rejected(tmp_path,monkeypatch):
    setup(monkeypatch)
    args=dict(pool=None,candidates=[dict(id='p')],opponents=[dict(id='f')],band=100,n=4,
              out=tmp_path/'one',cache=tmp_path/'cache',
              context=dict(plan_sha256='a'*64,runtime_sha256='b'*64,deterministic=True))
    module.evaluate_cached(**args)
    path=next((tmp_path/'cache').glob('*.json'));data=json.loads(path.read_text())
    data['record']['results'][0]['episodes'].pop()
    # Even a recomputed checksum cannot turn incomplete pairs into evidence.
    data['record_sha256']=module.digest(data['record']);path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='paired episodes'):
        module.evaluate_cached(**{**args,'out':tmp_path/'two'})
