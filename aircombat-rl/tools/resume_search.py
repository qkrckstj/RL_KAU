"""Configurable continuation for the autonomous experiment loop; frozen historical sources are unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule, FloatSchedule

from tools.plan_a import ROOT, evaluate, save_training_state, timestamp, write_json


def resize_replay(model, capacity):
    """Repack a plain one-env replay oldest-first without dropping experiences.

    Apply to BOTH comparison arms so initial sample indices have the same meaning.
    """
    from stable_baselines3.common.buffers import ReplayBuffer
    old = model.replay_buffer
    if type(old) is not ReplayBuffer or old.n_envs != 1 or old.optimize_memory_usage:
        raise ValueError("Only plain one-env non-optimized ReplayBuffer is supported")
    size = old.size()
    if capacity < size:
        raise ValueError("Capacity must preserve all existing experiences")
    new = ReplayBuffer(capacity, old.observation_space, old.action_space, device=old.device,
        n_envs=1, optimize_memory_usage=False, handle_timeout_termination=old.handle_timeout_termination)
    indices = (np.arange(size) + old.pos) % old.buffer_size if old.full else np.arange(size)
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        getattr(new, name)[:size] = getattr(old, name)[indices]
    new.full, new.pos = size == new.buffer_size, size % new.buffer_size
    model.replay_buffer, model.buffer_size = new, capacity
    return dict(old_capacity=old.buffer_size, new_capacity=capacity, retained=size,
                order="oldest-first; both arms repacked identically")


def resume(args):
    source, out = Path(args.run).resolve(), Path(args.out).resolve()
    checkpoint = source / "checkpoints" / f"step_{args.checkpoint}"
    if args.additional_steps <= 0 or args.additional_steps % 4:
        raise ValueError("Additional steps must be a positive multiple of four.")
    if args.val_every <= 0 or args.val_every % 4 or args.val_n < 2 or args.val_n % 2:
        raise ValueError("Use a validation interval divisible by four and a positive even match count.")
    if out.exists():
        raise FileExistsError("Use a new output directory; source checkpoints are never overwritten.")
    for name in ("policy_net.zip", "replay_buffer.pkl", "rng_state.pkl", "resume.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Complete learner/replay checkpoint required: {checkpoint / name}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    frozen_hash = hashlib.sha256((source / "utils.py").read_bytes()).hexdigest()
    expected_hash = config.get("source_sha256")
    if expected_hash is not None and frozen_hash != expected_hash:
        raise ValueError("Frozen preprocessing hash differs from the run config.")
    if expected_hash is None and (source / "utils.py").read_bytes() != (ROOT / "experiments/plan_a/core.py").read_bytes():
        raise ValueError("Cannot verify frozen preprocessing without source_sha256.")
    module_spec = importlib.util.spec_from_file_location("_resume_frozen_core", source / "utils.py")
    frozen_core = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(frozen_core)
    if config["spec"]["algorithm"] != "dqn":
        raise ValueError("This command resumes DQN only.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the matching PyTorch build/driver or choose cpu.")
    if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel XPU is unavailable. Install the matching build/driver or choose cpu.")
    torch.set_num_threads(1)
    env = Monitor(frozen_core.make_env(config["spec"], seed=args.seed))
    try:
        algorithm = frozen_core.DoubleDQN if config["spec"].get("double_dqn", False) else DQN
        model = algorithm.load(checkpoint / "policy_net.zip", env=env, device=args.device, force_reset=True)
        model.load_replay_buffer(checkpoint / "replay_buffer.pkl")
        previous_spec = dict(config["spec"])
        rate = getattr(args, "learning_rate", 0.0001)
        if not 0 < rate <= 0.001:
            raise ValueError("Expected learning rate in (0, 0.001]")
        model.learning_rate = rate
        model.lr_schedule = FloatSchedule(rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = rate
        config["spec"] = dict(config["spec"], learning_rate=rate)
        replay_resize = None
        if getattr(args, "replay_capacity", None) is not None:
            replay_resize = resize_replay(model, args.replay_capacity)
        if model.num_timesteps != args.checkpoint or abs(model.exploration_rate - json.loads((checkpoint / "resume.json").read_text(encoding="utf-8"))["exploration_rate"]) > 1e-8:
            raise ValueError("Saved step or exploration rate does not match its checkpoint metadata.")
        if model.n_steps != 1 or model.optimize_memory_usage:
            raise ValueError("Resume supports the saved one-step, non-memory-optimized replay configuration.")
        start_step, start_updates = model.num_timesteps, model._n_updates
        initial_replay_size = model.replay_buffer.size()
        model.set_random_seed(args.seed)
        # Only load these trusted, locally produced checkpoint files.
        with (checkpoint / "rng_state.pkl").open("rb") as stream:
            rng = pickle.load(stream)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if getattr(args, "rng_seed", None) is not None:
            model.set_random_seed(args.rng_seed)
        epsilon = getattr(args, "epsilon", 0.05)
        if not 0 <= epsilon <= 1:
            raise ValueError("Invalid epsilon")
        # Do not stretch or restart exploration when learn() changes its horizon.
        model.exploration_initial_eps = model.exploration_final_eps = epsilon
        model.exploration_fraction = 1.0
        model.exploration_schedule = LinearSchedule(epsilon, epsilon, 1.0)
        model.exploration_rate = epsilon
        out.mkdir(parents=True)
        for name in ("policy.py", "wrappers.py", "utils.py"):
            shutil.copyfile(source / name, out / name)
        wrapper = out / "wrappers.py"
        wrapper_text = wrapper.read_text(encoding="utf-8")
        old_spec_line = f"SPEC = {previous_spec!r}"
        if old_spec_line not in wrapper_text:
            raise ValueError("Cannot update exported learner specification")
        wrapper.write_text(wrapper_text.replace(old_spec_line, f"SPEC = {config['spec']!r}", 1), encoding="utf-8")
        metadata = dict(spec=config["spec"], source_sha256=frozen_hash,
                        replay_resize=replay_resize,
                        seed=args.seed, device=args.device,
                        torch=torch.__version__, start_step=start_step,
                        steps=start_step + args.additional_steps, additional_steps=args.additional_steps,
                        initial_gradient_updates=start_updates, initial_replay_size=initial_replay_size,
                        source_model_sha256=hashlib.sha256((checkpoint / "policy_net.zip").read_bytes()).hexdigest(),
                        val_band=getattr(args,"val_band",900000), val_n=args.val_n, val_every=args.val_every,
                        epsilon=epsilon, rng_seed=getattr(args,"rng_seed",None), started_utc=timestamp(),
                        continuation="Restored learner, optimizer, replay and CPU RNG; fresh episode; not exact trajectory resume.")
        write_json(out / "config.json", metadata)
        model.set_logger(configure(str(out / "metrics"), ["csv"]))
        started = time.perf_counter()

        class Progress(BaseCallback):
            def __init__(self):
                super().__init__()
                self.history, self.outcomes = [], Counter()
                self.best, self.best_step = (-1, -1e9), None
                self.next_eval, self.next_log = start_step + args.val_every, start_step + 10000

            def status(self, stage):
                write_json(out / "progress.json", dict(stage=stage, steps=model.num_timesteps,
                           target_steps=metadata["steps"], best_step=self.best_step,
                           best_validation_kills=self.best[0], train_outcomes=dict(self.outcomes),
                           elapsed_seconds=time.perf_counter() - started, updated_utc=timestamp()))

            def validate(self):
                self.status("validation")
                result = evaluate(model, config["spec"], getattr(args,"val_band",900000), args.val_n, frozen_core)
                result["step"] = model.num_timesteps
                if getattr(args, "replay_diagnostics", False):
                    replay = model.replay_buffer
                    size = replay.size()
                    indices = np.random.default_rng(1729).choice(size, min(size,2048), replace=False)
                    with torch.no_grad():
                        q = model.q_net(torch.as_tensor(replay.observations[indices,0],device=model.device))
                    result["replay_diagnostics"] = dict(size=size, capacity=replay.buffer_size,
                        positive_rewards=int((replay.rewards[:size] > 0).sum()),
                        terminal_transitions=int(replay.dones[:size].sum()), q_max=float(q.max()),
                        q_bound_violation_fraction=float(((q < -.20001)|(q > 1.00001)).float().mean()))
                summary = result["summary"]
                score = (summary["kills"], -(summary["t_kill"] if summary["t_kill"] is not None else 1e9))
                if score > self.best:
                    self.best, self.best_step = score, model.num_timesteps
                    model.save(out / "policy_net.zip")
                self.history.append(result)
                target = out / "checkpoints" / f"step_{model.num_timesteps}"
                target.mkdir(parents=True, exist_ok=True)
                model.save(target / "policy_net.zip")
                milestones = getattr(args, "replay_milestones", None)
                if milestones is None or model.num_timesteps in milestones:
                    save_training_state(model, target)
                write_json(out / "validation.json", self.history)
                self.status("training")
                print(f"VAL step={model.num_timesteps} wins={summary['kills']}/{args.val_n}", flush=True)

            def _on_training_start(self):
                # The existing policy remains eligible, so resuming cannot discard it.
                self.validate()

            def _on_rollout_start(self):
                if model.num_timesteps >= self.next_eval:
                    self.validate()
                    self.next_eval += args.val_every

            def _on_step(self):
                for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
                    if done:
                        self.outcomes[info.get("outcome", "unknown")] += 1
                if model.num_timesteps >= self.next_log:
                    self.status("training")
                    print(f"TRAIN {model.num_timesteps}/{metadata['steps']}", flush=True)
                    self.next_log += 10000
                return True

        callback = Progress()
        model.learn(total_timesteps=args.additional_steps, reset_num_timesteps=False, callback=callback)
        if not callback.history or callback.history[-1]["step"] != model.num_timesteps:
            callback.validate()
        assert model.num_timesteps == metadata["steps"]
        if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
            raise RuntimeError("Non-finite model parameters.")
        model.save(out / "final_net.zip")
        result = dict(status="complete", steps=model.num_timesteps, start_step=start_step,
                      gradient_updates_before=start_updates, gradient_updates_after=model._n_updates,
                      replay_size_before=initial_replay_size, replay_size_after=model.replay_buffer.size(),
                      best_step=callback.best_step, best_validation_kills=callback.best[0],
                      train_outcomes=dict(callback.outcomes), finished_utc=timestamp())
        write_json(out / "result.json", result)
        callback.status("complete")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--run", required=True, help="Training run containing frozen design and checkpoints/")
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--additional-steps", type=int, default=51200)
    parser.add_argument("--seed", type=int, default=10002, help="Seed for the new episode and device RNG")
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    parser.add_argument("--val-every", type=int, default=51200)
    parser.add_argument("--val-n", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(resume(args), indent=2))


if __name__ == "__main__":
    main()
