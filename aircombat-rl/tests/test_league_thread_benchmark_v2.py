from copy import deepcopy
import pytest
from tools.league_thread_benchmark_v2 import decision, SCENARIOS


def rows():
    records=[]
    for repeat in (0,1):
        for name,mode,n in SCENARIOS:
            records.append(dict(scenario=name,repeat=repeat,match_seconds=100/n,
                initialized_workers=[dict(private_gib=.3)]*n,
                resources=dict(minimum_commit_headroom_gib=4.,minimum_available_memory_gib=4.),
                results=[dict(own='a',foe='b',band=1,summary={},episodes=[dict(won=True)])]))
    return records


def test_measured_memory_and_two_repeats_required():
    records=rows()
    assert decision(records,[])['recommended_workers']==16
    records[-1]['resources']['minimum_commit_headroom_gib']=2.9
    assert decision(records,[])['recommended_workers']==12
    records=[r for r in records if not (r['scenario']=='single_w12' and r['repeat']==1)]
    assert decision(records,[])['recommended_workers']==8


def test_no_speed_claim_with_different_episodes_or_missing_baseline():
    records=rows();records[-1]['results']=deepcopy(records[-1]['results'])
    records[-1]['results'][0]['episodes'][0]['won']=False
    with pytest.raises(ValueError,match='episode'): decision(records,[])
    with pytest.raises(ValueError,match='incomplete'):
        decision([r for r in rows() if r['scenario']!='default_w2'],[])
