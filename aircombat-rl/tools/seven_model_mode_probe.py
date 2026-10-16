"""Explicit identity and input binding for a greedy deployment diagnostic.

The candidate is deterministic; both action keys still select the original
opponents' distinct sampled replicas. This never changes the training runner.
"""
from pathlib import Path

from experiments.rear_async.storage import identity, sha, verify
from tools.seven_model_league_v7 import model_identity, protocol, admission_allowed


def mode_specs(checkpoint):
    checkpoint = Path(checkpoint)
    verify(checkpoint)
    folder = checkpoint / 'greedy'
    if not (folder / 'policy.py').is_file() or not (folder / 'policy_net.zip').is_file():
        raise ValueError('Missing immutable greedy export')
    return {str(seed): dict(id=checkpoint.name + f'_greedy_a{seed}',
                           kind='submission', design=str(folder),
                           weights=str(folder / 'policy_net.zip'))
            for seed in (4900, 4901)}


def mode_inputs(checkpoint, pool):
    return {str(path): sha(path)
            for entry in [dict(sampled_specs=mode_specs(checkpoint)), *pool]
            for spec in entry['sampled_specs'].values()
            for path in Path(spec['design']).iterdir() if path.is_file()}


def mode_protocol(config, pool, checkpoint, stage, iteration):
    checkpoint = Path(checkpoint)
    if stage not in ('screen', 'confirmation') or config['smoke']:
        raise ValueError('Production screen or confirmation required')
    own = mode_specs(checkpoint)
    export = {str(p): sha(p) for p in (checkpoint / 'greedy').iterdir() if p.is_file()}
    result = protocol(config, pool, checkpoint, stage, iteration)
    result.update(candidate_mode='greedy', candidate_rng='unused_deterministic',
                  model_id=identity(dict(checkpoint_id=model_identity(checkpoint),
                                         mode='greedy', specs=own, export=export)))
    return result


def validate_mode_result(result, config, pool, checkpoint, stage, iteration):
    """Call on authenticated_result output, then bind it to intended inputs."""
    if result['protocol'] != mode_protocol(config, pool, checkpoint, stage, iteration):
        raise ValueError('Mode, identity or evaluation conditions differ')
    if result['input_hashes'] != mode_inputs(checkpoint, pool):
        raise ValueError('Actual greedy/original opponent input binding differs')


def admission_entry(screen, confirm, config, pool, checkpoint, iteration):
    """Build an entry from raw-authenticated results; caller reserves fresh ICs."""
    validate_mode_result(screen, config, pool, checkpoint, 'screen', iteration)
    validate_mode_result(confirm, config, pool, checkpoint, 'confirmation', iteration)
    if not admission_allowed(screen, confirm):
        raise ValueError('Both panels must pass with disjoint confirmation conditions')
    policy_id = screen['protocol']['model_id']
    entry = dict(id=Path(checkpoint).name + '_greedy_' + policy_id[:12],
                 checkpoint=str(checkpoint), mode='greedy', model_id=policy_id,
                 sampled_specs=mode_specs(checkpoint), original=False)
    if any(p['id'] == entry['id'] or p.get('model_id') == policy_id for p in pool):
        raise ValueError('Duplicate policy identity')
    return entry
