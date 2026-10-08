import pytest
from experiments.league.residual_env import CoverageSampler
from experiments.league.periodic_coverage import PeriodicCoverageSampler


def test_original_period_reproduces_entire_stream():
    old=CoverageSampler([.1,.3,.6],57)
    new=PeriodicCoverageSampler([.1,.3,.6],57,2)
    assert [old.next() for _ in range(600)]==[new.next() for _ in range(600)]
    assert old.coverage==new.coverage
    assert old.rng.bit_generator.state==new.rng.bit_generator.state


def test_quarter_coverage_even_when_weighted_draws_exclude_foes():
    sampler=PeriodicCoverageSampler([1.,0.,0.,0.,0.,0.,0.],91,4)
    rows=[sampler.next() for _ in range(28)]
    assert set(rows[::4])==set(range(7))
    assert len(set(rows[::4]))==len(rows[::4])
    assert all(rows[i]==0 for i in range(28) if i%4)


@pytest.mark.parametrize('period',[0,1,3,-1,True,2.0])
def test_rejects_unqualified_periods(period):
    with pytest.raises((TypeError,ValueError)):
        PeriodicCoverageSampler([1.,1.],1,period)
