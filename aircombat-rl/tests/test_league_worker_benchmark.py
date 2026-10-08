import pytest
from tools.league_worker_benchmark import choose


def test_choose_retains_memory_headroom_and_avoids_insignificant_extra_workers():
    rows = [dict(workers=w, seconds=s, minimum_free_memory_gib=free)
            for _ in range(2) for w, s, free in [(3, 30, 9), (6, 16, 7), (9, 15.8, 5), (12, 12, 1)]]
    decision = choose(rows)
    assert decision['workers'] == 6
    assert decision['speedup_vs_three'] == pytest.approx(30/16)


def test_choose_rejects_unrepeated_results():
    with pytest.raises(ValueError, match='No configuration'):
        choose([dict(workers=3, seconds=10, minimum_free_memory_gib=9)])
