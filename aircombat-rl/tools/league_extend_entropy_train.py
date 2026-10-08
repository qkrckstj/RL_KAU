"""Matched entropy-zero treatment using completed prior6 runs as controls."""
from argparse import ArgumentParser
from pathlib import Path
import copy, os, shutil
from tools import league_extend_teacher_train as engine
from tools import league_thread_benchmark as io
from tools.league_extend_teacher_analysis import analyze as verify_raw


def freeze(source, out, smoke, qualification):
    if out.exists():
        raise FileExistsError(out)
    if any(os.environ.get(k) != '1' for k in io.THREAD_KEYS):
        raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):
        raise RuntimeError('Control still live')
    old = io.read(source/'plan.json')
    io.verify(old)
    raw = io.read(source/'raw_extend_analysis.json')
    if raw['status'] != 'extend_teacher_raw_verified' or raw['training_steps'] != 5242880:
        raise ValueError('Completed verified controls required')
    for name, digest in raw['input_sha256'].items():
        if io.sha(io.ROOT/name) != digest:
            raise ValueError('Changed control evidence')
    sources = dict(old['source_sha256'])
    sources[io.relative(Path(__file__).resolve())] = io.sha(Path(__file__))
    inputs = dict(old['input_sha256'])
    for p in [source/'plan.json', source/'completion.json', source/'raw_extend_analysis.json']:
        inputs[io.relative(p)] = io.sha(p)
    if not smoke:
        q = io.read(qualification/'raw_extend_analysis.json')
        qp = io.read(qualification/'plan.json')
        io.verify(qp)
        if q['status'] != 'extend_teacher_raw_verified' or qp['ppo']['ent_coef'] != 0 or qp['source'] != io.relative(source):
            raise ValueError('Entropy-zero integration required')
        for name, digest in q['input_sha256'].items():
            if io.sha(io.ROOT/name) != digest:
                raise ValueError('Changed qualification')
        for p in [qualification/'plan.json', qualification/'raw_extend_analysis.json', qualification/'completion.json']:
            inputs[io.relative(p)] = io.sha(p)
    ppo = dict(old['ppo'], ent_coef=0.)
    seeds = [6798] if smoke else old['seeds']
    starts, initializations = {'prior6': {}}, {'prior6': {}}
    for seed in seeds:
        initial, evidence = engine.initial_model(out/f'initial/prior6/s{seed}', old['teacher'], seed, 6, ppo)
        if not smoke:
            control = old['initializations']['prior6'][str(seed)]
            assert evidence['parameter_digest'] == control['parameter_digest']
            a, b = dict(evidence['ppo_configuration']), dict(control['ppo_configuration'])
            assert a.pop('ent_coef') == 0. and b.pop('ent_coef') == .005 and a == b
        starts['prior6'][str(seed)] = initial
        initializations['prior6'][str(seed)] = evidence
    for p in (out/'initial').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            inputs[io.relative(p)] = io.sha(p)
    plan = copy.deepcopy(old)
    development = [next(s for s in old['opponents'] if s['id'] == k) for k in ['ace','evader','temporal_extend_left']] if smoke else old['development_opponents']
    plan.update(source=io.relative(source), smoke=smoke, ppo=ppo, seeds=seeds,
        arms=[dict(name='prior6', prior_bias=6)], warm_starts=starts, initializations=initializations,
        source_ppo_configuration=initializations['prior6'][str(seeds[0])]['ppo_configuration'],
        training_bands=[170390000] if smoke else old['training_bands'],
        development_opponents=development, development_band=170400000 if smoke else old['development_band'],
        development_n=2 if smoke else old['development_n'], chunk_steps=20480 if smoke else old['chunk_steps'],
        base_chunks=1, maximum_chunks=1 if smoke else 2, cached_baseline={},
        source_sha256=sources, input_sha256=inputs, environment=io.environment(),
        matched_control_source=io.relative(source), changed_ppo_setting=dict(ent_coef=[.005,0.]),
        quality_scope='Entropy-zero prior6 treatment; exact same initial parameter tensors, two learner RNGs, teacher, opponent sampler and training seed streams as completed prior6 controls. Trajectories may diverge. Stochastic action sampling remains enabled.',
        experiment_scope='Reused96M development conditions, NOT unseen. Existing controls are reused without retraining. Two independently initialized learners per treatment. No final/promotion/GitHub claim.')
    io.verify(plan)
    io.write(out/'plan.json', plan)
    for name in sources:
        p = out/'source_snapshot'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io.ROOT/name, p)
    return plan


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--qualification', type=Path, default=Path('runs/league_extend_entropy_smoke_20261007'))
    a = p.parse_args()
    engine.freeze = freeze
    engine.run(a.source.resolve(), a.out.resolve(), a.smoke, a.qualification.resolve())
    result = verify_raw(a.out.resolve())
    print('entropy-zero raw verified', result['training_steps'], flush=True)
