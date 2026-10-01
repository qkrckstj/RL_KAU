"""Diagnose completed A3 validation stability and final replay/Q values.

Read-only with respect to training artifacts; writes new diagnostic files.
Uses existing validation/test records and replay states, never new test seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from tools.plan_a import timestamp, write_json
from tools.plan_a_shaping import read, sha


def replay_diagnostics(run, budget):
    checkpoint = run / f"checkpoints/step_{budget}"
    model = DQN.load(checkpoint / "policy_net.zip", device="cpu")
    model.load_replay_buffer(checkpoint / "replay_buffer.pkl")
    replay = model.replay_buffer
    size = replay.size()
    indices = np.random.default_rng(1729).choice(size, min(2048, size), replace=False)
    obs = torch.as_tensor(replay.observations[indices, 0], device="cpu")
    next_obs = torch.as_tensor(replay.next_observations[indices, 0], device="cpu")
    actions = torch.as_tensor(replay.actions[indices, 0], dtype=torch.int64)
    rewards = replay.rewards[indices, 0]
    terminal = replay.dones[indices, 0] * (1 - replay.timeouts[indices, 0])
    with torch.no_grad():
        q = model.q_net(obs)
        greedy = q.argmax(dim=1)
        top = q.topk(2, dim=1).values
        chosen = q.gather(1, actions).flatten()
        targets = torch.as_tensor(rewards) + model.gamma * torch.as_tensor(1 - terminal) * model.q_net_target(next_obs).amax(dim=1)
        errors = chosen - targets
    config = read(run / "config.json")
    if sha(run / "utils.py") != config["source_sha256"]:
        raise ValueError("Frozen training source hash mismatch.")
    module_spec = importlib.util.spec_from_file_location("_diagnostic_frozen", run / "utils.py")
    frozen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(frozen)
    shaped = config["spec"].get("reward", "terminal") == "potential_v1"
    phi = np.asarray([frozen.potential(x) if shaped else 0.0 for x in obs.numpy()], dtype=np.float32)
    # One terminal reward lies in [-0.2, 1]. All preceding base rewards are 0.
    # Telescoping shaping shifts this bound by -Phi(current).
    lower = torch.as_tensor(-0.2 - phi)
    upper = torch.as_tensor(1.0 - phi)
    violations = (q < lower[:, None] - 1e-5) | (q > upper[:, None] + 1e-5)
    chosen_violations = (chosen < lower - 1e-5) | (chosen > upper + 1e-5)
    all_rewards = replay.rewards[:size, 0]
    all_obs = replay.observations[:size, 0]
    action_counts = Counter(int(a) for a in replay.actions[:size, 0].flatten())
    greedy_counts = Counter(int(a) for a in greedy)
    return dict(
        checkpoint_sha256=sha(checkpoint / "policy_net.zip"), replay_size=size,
        sampled_states=len(indices), sampling_seed=1729,
        rewards=dict(min=float(all_rewards.min()), max=float(all_rewards.max()),
                     mean=float(all_rewards.mean()),
                     nonzero_fraction=float(np.count_nonzero(all_rewards) / size)),
        terminal_count=int(replay.dones[:size, 0].sum()),
        external_timeout_count=int(replay.timeouts[:size, 0].sum()),
        training_action_counts=dict(sorted(action_counts.items())),
        greedy_action_counts_sample=dict(sorted(greedy_counts.items())),
        dominant_greedy_action_fraction=max(greedy_counts.values()) / len(indices),
        q_min=float(q.min()), q_max=float(q.max()),
        analytical_global_return_bounds=[-0.7, 1.5] if shaped else [-0.2, 1.0],
        state_bound_rule="[-0.2 - Phi(state), 1.0 - Phi(state)]",
        all_action_return_bound_violation_fraction=float(violations.float().mean()),
        sampled_action_return_bound_violation_fraction=float(chosen_violations.float().mean()),
        max_excess_above_state_return_bound=max(0.0, float((q - upper[:, None]).max())),
        mean_top_two_q_gap=float((top[:, 0] - top[:, 1]).mean()),
        td_rmse=float(errors.square().mean().sqrt()),
        td_mean=float(errors.mean()),
        own_weapon_zone_fraction=float(all_obs[:, 23].mean()),
        opponent_weapon_zone_fraction=float(all_obs[:, 24].mean()),
        limits="Off-policy replay diagnostics, not calibrated value accuracy or a causal explanation.",
    )


def diagnose(out):
    torch.set_num_threads(1)
    plan = read(out / "plan.json")
    summary = read(out / "summary.json")
    final_budget = plan["budgets"][-1]
    records = []
    for reward in plan["rewards"]:
        for seed in plan["seeds"]:
            run = out / reward / f"s{seed}"
            history = [r for r in read(run / "validation.json") if r["step"] > 0]
            result = read(run / "result.json")
            selected = next(c for c in read(out / "selection.json")["choices"]
                            if c["reward"] == reward and c["seed"] == seed
                            and c["budget"] == final_budget and c["kind"] == "validation_selected")
            outcomes = result["train_outcomes"]
            first_win = next((r["step"] for r in history if r["summary"]["kills"] > 0), None)
            last = history[-1]["summary"]
            records.append(dict(
                reward=reward, seed=seed, training_episodes=sum(outcomes.values()),
                training_wins=outcomes.get("kill", 0),
                training_win_rate=outcomes.get("kill", 0) / max(sum(outcomes.values()), 1),
                validation_first_nonzero_step=first_win,
                validation_nonzero_checkpoints=sum(r["summary"]["kills"] > 0 for r in history),
                validation_checkpoint_count=len(history),
                validation_wins_by_step=[dict(step=r["step"], wins=r["summary"]["kills"]) for r in history],
                selected_step=selected["step"], selected_validation=selected["validation"],
                final_validation=last,
                validation_win_drop=selected["validation"]["kills"] - last["kills"],
                final_replay=replay_diagnostics(run, final_budget),
            ))
    result = dict(created_utc=timestamp(), run=str(out),
                  decision=summary["decision"], records=records,
                  scope="Existing validation records and final replay; no new test queries.",
                  selected_before_test=read(out / "selection.json")["selected_policy"])
    write_json(out / "diagnostics.json", result)
    lines = [
        "# A3 후속 진단", "",
        "완료된 실험의 검증 이력과 마지막 replay/Q 값을 분석했다. 새로운 시험 집합은 열지 않았다.", "",
        "| 보상 | seed | 학습 승리/경기 | 승리 검증 체크포인트 | 선택/최종 검증 승리 | 최종 replay 최다 행동 비율 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(f"| {r['reward']} | {r['seed']} | {r['training_wins']}/{r['training_episodes']} "
                     f"| {r['validation_nonzero_checkpoints']}/{r['validation_checkpoint_count']} "
                     f"| {r['selected_validation']['kills']} / {r['final_validation']['kills']} "
                     f"| {r['final_replay']['dominant_greedy_action_fraction']:.1%} |")
    lines += [
        "", "최다 행동 비율은 마지막 replay에서 고정 표본 2048개를 뽑아 현재 greedy 정책으로 계산했다.",
        "상태 분포가 제한될 수 있으므로 높은 비율만으로 정책 붕괴의 원인을 확정하지 않는다.",
        "TD 오차는 같은 learner/target network에 대한 내부 적합도이며 실제 승률을 대신하지 않는다.",
        "shaping 적용 후 비영 보상이 많아지는 것과 공식 승률 향상은 구분한다.", "",
        "추가로 가능한 할인 누적 보상의 해석적 범위를 확인했다. 종료 보상은 [-0.2, 1],",
        "potential 보상은 전체적으로 [-0.7, 1.5] 이내이며 상태별로 [-0.2-Phi, 1-Phi]로 더 좁힐 수 있다.",
        "이 범위를 벗어난 Q 예측은 가치 추정 오차의 직접적인 증거다. TD 오차가 작다고",
        "실제 가치가 정확한 것은 아니다. 과대 추정이 성능 하락의 유일한 원인임을 입증한 것은 아니다.", "",
        "사전에 고정한 후속 판단:", "",
        "~~~json", json.dumps(summary["decision"], indent=2), "~~~", "",
    ]
    (out / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(result["decision"], indent=2), flush=True)
    for r in records:
        print(f"{r['reward']}/s{r['seed']}: train {r['training_wins']}/{r['training_episodes']}, "
              f"validation retained {r['final_validation']['kills']}/{r['selected_validation']['kills']}, "
              f"dominant action {r['final_replay']['dominant_greedy_action_fraction']:.1%}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Completed A3 experiment directory")
    args = parser.parse_args()
    diagnose(Path(args.out).resolve())


if __name__ == "__main__":
    main()
