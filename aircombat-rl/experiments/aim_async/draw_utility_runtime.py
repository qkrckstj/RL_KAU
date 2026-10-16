"""Recover the pending bounded extension decision before resumed training."""
from experiments.rear_async.storage import read, write, verify


def recover_minimum_decision(out, seed, meta, plan, request):
    path = out/f's{seed}_budget_decision.json'
    if path.exists():
        return read(path)
    if meta is None or plan['smoke'] or meta['additional_steps'] < plan['minimum_steps']:
        return None
    if meta['additional_steps'] > plan['minimum_steps']:
        raise ValueError('Missing bounded extension decision above minimum; preserve and diagnose')
    checkpoint = out/'checkpoints'/f"s{seed}_step_{plan['minimum_steps']:09d}"
    stored = verify(checkpoint)
    if (stored['seed'], stored['additional_steps'], stored['config_id']) != (seed, plan['minimum_steps'], plan['config_id']):
        raise ValueError('Resumed minimum checkpoint identity differs')
    # An interruption can occur between publication and either queue request.
    # request() is idempotent: completed cells keep their original identities.
    request(out, checkpoint, 'monitor')
    request(out, checkpoint, 'development')
    from tools.aim_cpu_cycle_audit import wait_for_seed_evaluations
    from experiments.aim_async.judgement import budget_decision
    wait_for_seed_evaluations(out, seed)
    completed = [read(p) for p in (out/'evaluations').glob('*/completion.json')]
    points = [e for e in completed if not e['baseline'] and e['kind'] == 'monitor' and e['seed'] == seed]
    broad = [e for e in completed if not e['baseline'] and e['kind'] == 'development'
             and e['seed'] == seed and e['additional_steps'] == plan['minimum_steps']]
    monitor_base = [e for e in completed if e['baseline'] and e['kind'] == 'monitor']
    broad_base = [e for e in completed if e['baseline'] and e['kind'] == 'development']
    if len(points) < 3 or len(broad) != 1 or len(monitor_base) != 1 or len(broad_base) != 1:
        raise ValueError('Required paired budget evaluations missing or ambiguous')
    decision = budget_decision(points, monitor_base[0]['summary'], plan['minimum_steps'],
        plan['minimum_steps'], plan['maximum_steps'], broad=broad[0]['summary'], broad_baseline=broad_base[0]['summary'])
    write(path, decision)
    return decision
