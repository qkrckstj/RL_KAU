"""Actual-flight parity of the preserved teacher, export, and PPO wrapper.

Reused compatibility conditions only. Not a policy performance experiment.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import time
import traceback

from tools.league_thread_benchmark import (
    ROOT, read, write, sha, relative, verify, resources, environment, THREAD_KEYS)


class Trace:
    def __init__(self, env, training=False):
        self.env, self.training = env, training
        self._combat = env.unwrapped._combat
        self.digest = hashlib.sha256()

    def add(self, action=None, reward=None, flags=None):
        import numpy as np
        base = self.env.unwrapped
        raw = base._obs(base._last)
        self.digest.update(np.asarray(raw, dtype='<f4').tobytes())
        # Includes both aircraft's HP, track, geometry and complete weapon data.
        self.digest.update(json.dumps(base._last, sort_keys=True, default=float).encode())
        self.digest.update(json.dumps([action, reward, flags]).encode())

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.add()
        return result

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if self.training:
            flags = info['official_episode_flags']
            term, trunc = flags['terminated'], flags['truncated']
            reward = info['official_reward']
        self.add(int(action), float(reward), [bool(term), bool(trunc)])
        return obs, reward, term, trunc, info


def job(case, teacher, exported, policy_path):
    import numpy as np
    import torch
    from tools.grade import play
    from tools.league_matches import MatchEnv
    from experiments.league.residual_env import ArchiveFight
    from experiments.league.residual_worker import LeagueWorker
    from experiments.league.residual_ppo_policy import ResidualPriorPolicy
    torch.set_num_threads(1)
    begun = time.perf_counter()
    answer = {}
    for name, spec in [('teacher', teacher), ('exported', exported)]:
        env = MatchEnv(spec, case['foe'], case['seat'])
        try:
            trace = Trace(env)
            rows = play(trace, env.own_action, case['seed'], 1, case['seat'])
            answer[name] = dict(episodes=rows, trajectory_sha256=trace.digest.hexdigest())
        finally:
            env.close()
    env = LeagueWorker(ArchiveFight(teacher, [case['foe']], [1.], 3200,
                      case['seed'], 0, 1, case['seat']))
    model = ResidualPriorPolicy.load(ROOT/policy_path, device='cpu')
    calls = 0
    evaluation = []
    def act(state):
        nonlocal calls
        calls += 1
        if case.get('interrupt_evaluation') and calls == 17:
            base = env.unwrapped
            def snapshot():
                return pickle.dumps((base.episode_index, base.episode_metadata,
                    base.sampler.__dict__, base.np_random.bit_generator.state,
                    env.previous_potential, base._last))
            before = snapshot()
            rng_before = (pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
            result = env.evaluate_partition(exported, [case['foe']], 170000010, 2)
            if before != snapshot() or rng_before[0] != pickle.dumps(np.random.get_state()) or not torch.equal(rng_before[1],torch.get_rng_state()):
                raise AssertionError('Evaluation changed paused training state/RNG')
            evaluation.extend(result)
        action, _ = model.predict(state, deterministic=True)
        return int(action)
    try:
        trace = Trace(env, training=True)
        rows = play(trace, act, case['seed'], 1, case['seat'])
        answer['training'] = dict(episodes=rows, trajectory_sha256=trace.digest.hexdigest())
        if not answer['teacher'] == answer['exported'] == answer['training']:
            raise AssertionError(f'Actual-flight identity failed: {case}')
        return dict(case=case, paths=answer, equality=True, paused_evaluation=evaluation,
                    elapsed_seconds=time.perf_counter()-begun)
    finally:
        env.close()


def freeze(out):
    if out.exists(): raise FileExistsError('Use a fresh qualification folder')
    if any(os.environ.get(k) != '1' for k in THREAD_KEYS):
        raise ValueError('Set thread limits before Python starts')
    previous = ROOT/'runs/league_pursuit_context_train_20261007'
    old = read(previous/'plan.json'); verify(old)
    if read(previous/'completion.json')['status'] != 'repair_profile_failed':
        raise ValueError('Unexpected predecessor state')
    benchmark = ROOT/'runs/league_flexible_benchmark_20261007'
    if read(benchmark/'completion.json')['status'] != 'benchmark_complete':
        raise ValueError('Execution comparison is not complete')
    proto = ROOT/'runs/residual_policy_prototype_20261007'
    exported = read(proto/'initial_model/entrant.json')
    opponents = old['opponents']
    wanted = ['ace','lead','pursuit','temporal_extend_right','temporal_weave_right',
              old['warm_start']['id'],'ddqn_s0',old['unrestricted_reference']['id']]
    by_id = {s['id']:s for s in opponents}
    cases = [dict(foe=by_id[name], seat=seat, seed=170000000+offset,
                  interrupt_evaluation=name=='ace' and offset==0)
             for name in wanted for seat in ('red','blue') for offset in range(2)]
    sources = dict(old['source_sha256']); inputs = dict(old['input_sha256'])
    for name in ('experiments/league/residual_features.py','experiments/league/residual_env.py',
                 'experiments/league/residual_ppo_policy.py','experiments/league/residual_worker.py',
                 'tools/league_residual_export.py','tools/league_residual_qualify.py'):
        sources[name] = sha(ROOT/name)
    paths = [previous/'completion.json', previous/'completed_experiment_analysis.json',
             benchmark/'completion.json', proto/'initial_actor_critic.pt']
    paths += list((proto/'initial_model').glob('*'))
    for path in paths:
        if path.is_file(): inputs[relative(path)] = sha(path)
    plan = dict(teacher=old['warm_start'],exported=exported,cases=cases,
        policy_path=relative(proto/'initial_actor_critic.pt'),workers=3,
        source_sha256=sources,input_sha256=inputs,environment=environment(),
        scope='96 whole flights plus4 paused-worker evaluation flights. Reused compatibility ICs. Raw observations/actions/full combat state/official rewards and flags exact. No learned-policy performance claim.')
    verify(plan); write(out/'plan.json',plan)
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    return plan


def main(out):
    plan=freeze(out)
    before=resources()
    if before['commit_headroom_gib']<4.5: raise RuntimeError('Insufficient commit headroom')
    write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=before))
    records=[]
    with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(job,c,plan['teacher'],plan['exported'],plan['policy_path']):i
                 for i,c in enumerate(plan['cases'])}
        for future in as_completed(futures):
            result=future.result();records.append(result)
            write(out/f"case_{futures[future]:03d}.json",result)
            print(f'Qualified {len(records)}/{len(futures)} cases',flush=True)
    verify(plan)
    write(out/'completion.json',dict(status='actual_flight_qualification_passed',cases=len(records),
        identity_flights=3*len(records),evaluation_flights=sum(sum(len(r['episodes']) for r in c['paused_evaluation']) for c in records),
        exact_trajectories=True,evaluation_preserves_paused_training=True,
        source_sha256=sha(Path(__file__)),resources_after=resources(),scope=plan['scope']))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try: main(args.out.resolve())
    except BaseException:
        write(args.out/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid()))
        raise
