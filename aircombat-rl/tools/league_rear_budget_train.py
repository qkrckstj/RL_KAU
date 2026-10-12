"""Matched curriculum-only continuation with explicit RNG and worker configuration checks."""
import gc,time,random
import numpy as np
from functools import partial
from tools import league_thread_benchmark as io
from tools.league_health_round import RULE

def train(plan,warm,seed,band,folder):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from experiments.league.grouped_vec_env import GroupedSubprocVecEnv
    from experiments.league.tail_actual_reward import make_worker as make_health_worker
    from tools.league_residual_recover import load_continuation
    from tools.league_residual_train import save_checkpoint
    from tools.league_sampled_strategy import make_progress
    from tools.league_residual_action_mode import build_variant
    from tools.league_history_clone_matched_check import ppo_configuration
    from tools.league_sampled_ppo_continue import same_tree
    from tools.league_residual_profile import tensor_digest
    folder.mkdir();torch.set_num_threads(1);local=dict(plan,seed=seed,ic_band=band);vec=None;model=None
    if io.resources()['commit_headroom_gib']<4.5:raise MemoryError('Learning startup headroom')
    try:
        vec=GroupedSubprocVecEnv([partial(make_health_worker,local,i) for i in range(plan['workers'])],plan['processes'])
        io.write(folder/'initialized_workers.json',dict(workers=vec.env_method('profile_status'),resources=io.resources()))
        if io.resources()['commit_headroom_gib']<2.8:raise MemoryError('Learning initialized headroom')
        model,prefix=load_continuation(io.ROOT/warm['learner'],vec,seed,plan['ppo']['n_steps'])
        assert ppo_configuration(model)==plan['source_ppo_configuration']
        before=PPO.load(io.ROOT/warm['learner'],device='cpu')
        assert same_tree(before.policy.state_dict(),model.policy.state_dict()) and same_tree(before.policy.optimizer.state_dict(),model.policy.optimizer.state_dict())
        del before;gc.collect();initial=tensor_digest(model.policy.state_dict())
        original_bias=model.policy.prior_bias;original_gamma=model.gamma
        assert original_bias==6. and original_gamma==.999
        gamma=plan['ppo']['gamma'];assert gamma in (.999,.9997)
        model.gamma=gamma;model.rollout_buffer.gamma=gamma
        worker_gammas=vec.get_attr('gamma');assert worker_gammas==[gamma]*plan['workers']
        worker_curricula=vec.get_attr('curriculum');assert worker_curricula==[plan['curriculum']]*plan['workers']
        worker_rear_coefficients=vec.get_attr('rear_damage_coefficient');assert worker_rear_coefficients==[plan['rear_damage_coefficient']]*plan['workers']
        assert tensor_digest(model.policy.state_dict())==initial
        assert ppo_configuration(model)==dict(plan['source_ppo_configuration'],gamma=gamma)
        # Loading the verification copy resets process-global RNGs to its
        # stored seed. Restore the declared learner RNG before any rollout.
        model.set_random_seed(seed)
        learner_initial_seed=torch.initial_seed()
        assert learner_initial_seed==model.seed==seed
        model.set_logger(configure(str(folder/'metrics'),['csv']));progress=make_progress(folder,seed)
        history=[];elapsed=0.
        for block in range(plan['chunks']):
            before_steps=model.num_timesteps;before_games=progress.completed
            started=time.perf_counter();model.learn(plan['chunk_steps'],callback=progress,reset_num_timesteps=False);seconds=time.perf_counter()-started;elapsed+=seconds
            assert model.num_timesteps-before_steps==plan['chunk_steps']
            ck=folder/f'checkpoint_{block+1}';identity=folder.parent.name+'_'+folder.name+f'_c{block+1}'
            rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
            last=model._last_obs.copy();starts=model._last_episode_starts.copy()
            env_indices=vec.get_attr('episode_index')
            greedy=save_checkpoint(model,ck,plan['teacher'],identity)
            replicas=[dict(action_seed=a,spec=build_variant(greedy,ck/f'sampled_{a}',a,identity+f'_a{a}')) for a in (4900,4901)]
            # Export checks must not change the continuing learner's RNG or episode state.
            torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
            assert np.array_equal(model._last_obs,last) and np.array_equal(model._last_episode_starts,starts)
            assert vec.get_attr('episode_index')==env_indices
            row=dict(block=block+1,additional_steps=model.num_timesteps,total_since_original_parent=prefix+model.num_timesteps,chunk_learning_seconds=seconds,training_seconds=elapsed,chunk_training_games=progress.completed-before_games,training_games=progress.completed,learner=io.relative(ck/'learner.zip'),greedy=greedy,replicas=replicas,checkpoint_preserves_live_episode_and_rng=True)
            history.append(row);io.write(folder/f'chunk_{block+1}.json',row)
        restored=PPO.load(ck/'learner.zip',device='cpu');assert same_tree(restored.policy.optimizer.state_dict(),model.policy.optimizer.state_dict())
        assert restored.policy.prior_bias==model.policy.prior_bias==6.
        assert restored.gamma==restored.rollout_buffer.gamma==model.gamma==gamma
        result=dict(history=history,prefix_steps=prefix,worker_rear_coefficients=worker_rear_coefficients,worker_curricula=worker_curricula,original_gamma=original_gamma,applied_gamma=gamma,worker_gammas=worker_gammas,actual_learner_seed=learner_initial_seed,rule=RULE,seed=seed,learner=io.relative(ck/'learner.zip'),greedy=greedy,replicas=replicas,steps=model.num_timesteps,
                    learning_seconds=elapsed,training_episodes=progress.completed,training_outcomes=progress.counts,
                    initial_parameters=initial,final_parameters=tensor_digest(model.policy.state_dict()),minimum_commit_headroom_gib=progress.minimum_commit,
                    optimizer_restore_exact=True)
        assert result['initial_parameters']!=result['final_parameters']
        io.write(folder/'result.json',result);return result
    except BaseException:
        if model is not None:model.save(folder/'interrupted_learner.zip')
        raise
    finally:
        if vec is not None:vec.close()

