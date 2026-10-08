"""Distinct normalized CEM candidates and elites for subsequent experiments.

Not imported into already frozen runs. Exact duplicate anchors keep an index
mapping but do not occupy multiple candidate or elite slots.
"""
import numpy as np


def distinct_elites(order, population, count):
    population = np.asarray(population, dtype=float)
    if population.ndim != 2 or not np.isfinite(population).all() or count < 1:
        raise ValueError('Expected finite candidate matrix and positive elite count')
    chosen, seen = [], set()
    for index in order:
        key = tuple(population[index])
        if key not in seen:
            seen.add(key)
            chosen.append(int(index))
            if len(chosen) == count:
                return chosen
    raise ValueError('Not enough distinct candidates for the requested elite count')


def sample_population(rng, mean, std, anchors, size, uniform=2):
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    if (mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std < 0) or np.any(mean < 0) or np.any(mean > 1)
            or size < 1 or uniform < 0):
        raise ValueError('Invalid normalized CEM distribution or budget')
    population, seen, anchor_indices = [], {}, []

    def add(vector):
        key = tuple(vector)
        if key in seen:
            return seen[key], False
        index = len(population)
        seen[key] = index
        population.append(vector.copy())
        return index, True

    for anchor in anchors:
        vector = np.asarray(anchor, float)
        if (vector.shape != mean.shape or not np.isfinite(vector).all()
                or np.any(vector < 0) or np.any(vector > 1)):
            raise ValueError('Anchor outside normalized search space')
        index, _ = add(vector)
        anchor_indices.append(index)
    if len(population)+uniform > size:
        raise ValueError('Budget cannot fit distinct anchors and uniform immigrants')
    misses = 0
    while len(population) < size:
        # A zero-variance or boundary-clipped distribution must not loop forever.
        global_draw = len(population) >= size-uniform or misses >= 32
        vector = rng.random(mean.size) if global_draw else np.clip(rng.normal(mean, std), 0, 1)
        _, added = add(vector)
        misses = 0 if added else misses+1
    return np.asarray(population), anchor_indices
