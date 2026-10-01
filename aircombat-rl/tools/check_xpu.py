"""Verify Intel GPU training and CPU checkpoint portability on a project baseline.

Run from the repository root: python -m tools.check_xpu
This is a short compatibility check, not a policy performance evaluation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "project_01_circular", "project_02_evader",
    "project_03_advantaged", "project_04_fair",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=PROJECTS, default=PROJECTS[0])
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU unavailable: check the Python environment and Intel driver.")

    print(f"Python {platform.python_version()} / PyTorch {torch.__version__}", flush=True)
    print(f"GPU: {torch.xpu.get_device_name(0)}", flush=True)
    sys.path.insert(0, str(ROOT / "templates" / args.project / "baseline"))
    from policy import Policy
    from wrappers import make_env

    model = Policy.make_learner(make_env, seed=0, device="xpu")
    env = model.get_env()
    raw_env = None
    try:
        if args.steps <= model.learning_starts:
            raise ValueError(f"--steps must exceed learning_starts={model.learning_starts}.")
        if any(p.device.type != "xpu" for p in model.policy.parameters()):
            raise RuntimeError("The policy parameters are not all on XPU.")
        before = [p.detach().cpu().clone() for p in model.q_net.parameters()]
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, progress_bar=False)
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start
        changed = any(
            not torch.equal(old, new.detach().cpu())
            for old, new in zip(before, model.q_net.parameters())
        )
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        if not changed or not finite or model._n_updates < 1:
            raise RuntimeError("Expected finite, updated model weights after training.")

        raw_env = make_env(seed=123, shaped=False)
        obs, _ = raw_env.reset(seed=123)
        live = Policy(model=model)
        action = live.act(obs)
        if not raw_env.action_space.contains(action):
            raise RuntimeError("The policy returned an invalid action.")
        raw_env.step(action)
        with tempfile.TemporaryDirectory(prefix="aircombat-xpu-check-") as tmp:
            checkpoint = Path(tmp) / "policy_net.zip"
            model.save(checkpoint)
            restored_xpu = Policy(checkpoint, device="xpu")
            restored_cpu = Policy(checkpoint, device="cpu")
            x = torch.from_numpy(live.state(obs)).unsqueeze(0)
            with torch.no_grad():
                expected = model.q_net(x.to("xpu")).cpu().numpy()
                got_xpu = restored_xpu.model.q_net(x.to("xpu")).cpu().numpy()
                got_cpu = restored_cpu.model.q_net(x).numpy()
            np.testing.assert_allclose(got_xpu, expected, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(got_cpu, expected, rtol=1e-4, atol=1e-5)
            for restored in (restored_xpu, restored_cpu):
                if not raw_env.action_space.contains(restored.act(obs)):
                    raise RuntimeError("Reloaded policy returned an invalid action.")

        report = {
            "status": "passed",
            "project": args.project,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.xpu.get_device_name(0),
            "device": str(model.device),
            "steps": model.num_timesteps,
            "gradient_updates": model._n_updates,
            "weights_changed": changed,
            "weights_finite": finite,
            "checkpoint_reload_xpu_and_cpu": True,
            "training_seconds": round(elapsed, 3),
            "note": "Compatibility check only; not a performance or win-rate benchmark.",
        }
        out = ROOT / "runs" / "xpu_smoke" / args.project
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        if env is not None:
            env.close()
        if raw_env is not None:
            raw_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
