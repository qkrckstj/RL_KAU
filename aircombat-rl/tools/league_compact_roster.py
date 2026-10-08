"""Select a bounded active roster while preserving the full archived league."""
from collections import Counter, defaultdict
import hashlib
import json
import zipfile
from tools import league_thread_benchmark as io
from tools.league_tournament_metrics import code_groups

CORE = ['cem_original', 'ace', 'evader', 'fixed_left', 'fixed_right',
        'intercept_s2500_g3_c6', 'extend_cem_s6800_g3_c9',
        'control_s7400_t2621440_a4900', 'control_s7601_t1310720_a4900',
        'temporal_extend_left', 'temporal_delayed_left', 'temporal_extend_right',
        'novel_opening_right', 'league_contextual_speed_pilot_20261007_c0',
        'league_contextual_speed_pilot_20261007_c1']


def select(archive, estimates, previous=None):
    groups = code_groups(archive)
    by_id = {s['id']: s for s in archive}
    assert len(by_id) == len(archive) and set(CORE) <= set(by_id)
    buckets = defaultdict(list)
    evidence = {}
    for s in archive:
        key = s['id']; fingerprint = key
        if s['kind'] == 'submission' and zipfile.is_zipfile(io.ROOT/s['weights']):
            with zipfile.ZipFile(io.ROOT/s['weights']) as z:
                names = set(z.namelist())
                expected = {'parameters.json', 'residual_actor.npz', 'residual_metadata.json', 'sampling_metadata.json'}
                if names == expected:
                    meta = json.loads(z.read('sampling_metadata.json'))
                    assert meta['temperature'] == 1.0
                    hashes = {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(names-{'sampling_metadata.json'})}
                    # These verified policy sources use only temperature/action_seed from sampling metadata.
                    policy = (io.ROOT/s['design']/'policy.py').read_text(encoding='utf-8')
                    assert "self.action_seed = metadata['action_seed']" in policy
                    assert "metadata['temperature'] != 1.0" in policy
                    fingerprint = groups[key] + ':' + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                    evidence[key] = dict(source_group=groups[key], member_sha256=hashes, action_seed=meta['action_seed'])
        buckets[fingerprint].append(key)
    representatives = []
    aliases = {}
    for members in buckets.values():
        core = [k for k in members if k in CORE]
        assert len(core) <= 1
        chosen = core[0] if core else min(members)
        representatives.append(chosen)
        if len(members) > 1: aliases[chosen] = sorted(members)
    eligible = set(representatives)
    chosen = list(CORE)
    counts = Counter(groups[k] for k in chosen)
    def weakness(k): return (estimates[k]['win_estimate'], k)
    def add(k): chosen.append(k); counts[groups[k]] += 1
    for group in sorted(set(groups.values())):
        if not counts[group]: add(min((k for k in eligible if groups[k] == group), key=weakness))
    while len(chosen) < 48:
        remaining = eligible-set(chosen)
        valid = [k for k in remaining if counts[groups[k]] < 4]
        if not valid: raise ValueError('Cannot fill48 under group cap4')
        add(min(valid, key=lambda k: (counts[groups[k]], *weakness(k))))
    rotation = []
    if previous:
        # Rotate at most8 within the same code group, keeping core and coverage.
        # Previous evidence remains consumed; no unseen-opponent claim.
        old = previous['selected_ids']
        assert len(old) == 48 and set(old) <= eligible and set(CORE) <= set(old)
        chosen = list(old); counts = Counter(groups[k] for k in chosen)
        for incoming in sorted(eligible-set(old), key=weakness):
            outgoing = [k for k in chosen if k not in CORE and k in old and groups[k] == groups[incoming]]
            if not outgoing: continue
            remove = max(outgoing, key=weakness)
            chosen[chosen.index(remove)] = incoming
            rotation.append(dict(removed=remove, added=incoming))
            if len(rotation) == 8: break
    assert len(chosen) == len(set(chosen)) == 48
    assert set(groups[k] for k in chosen) == set(groups.values())
    assert max(Counter(groups[k] for k in chosen).values()) <= 4
    manifest = dict(archive_count=len(archive), eligible_after_rng_dedup=len(eligible), active_count=48,
        code_groups=len(set(groups.values())), selected_ids=chosen, core_ids=CORE,
        group_counts=dict(Counter(groups[k] for k in chosen)), rng_aliases=aliases,
        rng_equivalence_evidence=evidence, rotation=rotation,
        rule='Core plus at least one per code group; fill least represented groups with weakest measured candidates; cap4/group. Only byte-identical actor/teacher artifacts under identical policy sources collapse fixed action-RNG replicas. Code equality alone is not behavioral equality.',
        rotation_rule='Between completed frozen experiments only: pass previous manifest to rotate up to8 non-core representatives within their code group. Keep48 and all groups. No in-flight changes.',
        scope='Active training selection, not deletion or proof of behavioral coverage.109M/110M evidence consumed. Full archive remains available for periodic wider evaluation.')
    return [by_id[k] for k in chosen], manifest
