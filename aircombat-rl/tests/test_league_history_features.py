import numpy as np
import pytest
from experiments.league.history_features import HistoryFeatures,HISTORY_DIM,PUBLIC_DIM


def frame(value):
    x=np.zeros(PUBLIC_DIM+9,dtype=np.float32);x[0]=value;x[-8]=1
    return x


def test_causal_lags_and_initial_padding():
    h=HistoryFeatures();first=h.update(frame(0),0)
    assert first.shape==(HISTORY_DIM,)
    assert np.count_nonzero(first[PUBLIC_DIM:-9])==0
    h.update(frame(.2),1)
    h.update(frame(.4),2)
    out=h.update(frame(1),6)
    assert out[PUBLIC_DIM]==pytest.approx(.3)
    assert out[2*PUBLIC_DIM]==pytest.approx(.4)
    np.testing.assert_array_equal(out[-9:],frame(1)[-9:])


def test_reset_prevents_previous_opponent_history():
    h=HistoryFeatures();h.update(frame(-1),0);h.update(frame(1),5)
    h.reset();out=h.update(frame(.5),0)
    assert np.count_nonzero(out[PUBLIC_DIM:-9])==0


def test_clock_reversal_and_duplicate_clock():
    h=HistoryFeatures();h.update(frame(0),1)
    for _ in range(100):h.update(frame(.2),1)
    assert len(h.samples)==1
    with pytest.raises(ValueError):h.update(frame(0),0)


def test_history_is_owned_and_bounded():
    h=HistoryFeatures();x=frame(.4);h.update(x,0);x[0]=1
    out=h.update(frame(.6),1)
    assert out[PUBLIC_DIM]==pytest.approx(.1)
    for i in range(2,100):h.update(frame(.6),i)
    assert len(h.samples)<=6


@pytest.mark.parametrize('clock',[np.nan,np.inf])
def test_nonfinite_clock_rejected(clock):
    with pytest.raises(ValueError):HistoryFeatures().update(frame(0),clock)
