import numpy as np
import pytest
from tools.league_cem_population import distinct_elites, sample_population


def test_repeated_incumbent_mean_and_seed_do_not_fill_three_elite_slots():
    population = np.array([[.5,.5], [.5,.5], [.5,.5], [.1,.2], [.9,.8]])
    indices = distinct_elites(range(5), population, 3)
    assert indices == [0,3,4]
    assert np.all(population[indices].std(axis=0) > 0)
    with pytest.raises(ValueError, match='Not enough distinct'):
        distinct_elites(range(3), population, 3)


def test_duplicate_anchors_keep_mapping_but_free_slots_for_exploration():
    anchors = [[0,0], [.5,.5], [.5,.5], [.5,.5], [1,1]]
    args = ([.5,.5], [.2,.2], anchors, 10)
    a, mapping = sample_population(np.random.default_rng(8), *args)
    b, again = sample_population(np.random.default_rng(8), *args)
    assert mapping == again == [0,1,1,1,2]
    assert a.shape == (10,2) and len({tuple(v) for v in a}) == 10
    assert np.array_equal(a,b)
    assert np.all((a >= 0) & (a <= 1))
    for anchor, index in zip(anchors,mapping):
        assert np.array_equal(a[index],anchor)


def test_collapsed_distribution_still_fills_population_without_duplicates():
    population, _ = sample_population(np.random.default_rng(9), [0,0], [0,0], [[0,0]], 6)
    assert len({tuple(v) for v in population}) == 6
    with pytest.raises(ValueError, match='Budget cannot fit'):
        sample_population(np.random.default_rng(9), [0,0], [0,0], [[0,0]], 2, uniform=2)
