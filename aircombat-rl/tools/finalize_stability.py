"""Wait for an already-running comparison, then produce its review artifacts.

Does not start or extend training and never selects models from test scores.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.plan_a import ROOT, timestamp, write_json
from tools.plan_a_shaping import read


def finalize(out, plot_python):
    plan, summary = read(out / "plan.json"), read(out / "summary.json")
    selected = summary["selected_policy"]
    source = out / selected["reward"] / f"s{selected['seed']}"
    test = read(source / f"checkpoints/step_{selected['step']}/test_band_{plan['test_band']}.json")
    write_json(out / "selected_policy/test.json", test)
    subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.stability_visuals", "capture", "--out", str(out)],
        cwd=ROOT, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    subprocess.run([plot_python, "-X", "utf8", "-m", "tools.stability_visuals", "plot", "--out", str(out)],
        cwd=ROOT, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    groups = [g for g in summary["groups"] if g["budget"] == plan["budgets"][-1]]
    lines = ["# 학습률 비교 결과 — 2026-10-05", "",
        "기존 종료 보상·31개 관측·DQN을 유지하고 학습률만 비교했다.",
        f"각 조건 {len(plan['seeds'])}개 학습 시드 × {plan['budgets'][-1]:,}스텝. 같은 노트북 CPU에서 새로 학습했다.", "",
        f"| 학습률 | 모델 선택 | 시드별 승리 / 각 {plan['test_n']}경기 | 평균 승률 |",
        "|---|---|---|---:|"]
    for g in groups:
        label = "검증 최고" if g["kind"] == "validation_selected" else "마지막 모델"
        lines.append(f"| {plan['learning_rates'][g['condition']]} | {label} | {g['wins']} | {g['mean_win_rate']:.2%} |")
    diags = [d for d in summary["diagnostics"] if d["condition"] == "lr_1e4"]
    last = [d["late_validation_wins"] for d in diags]
    prev = [d["previous_validation_wins"] for d in diags]
    enough = len(diags) == 3 and all(len(x) == 5 for x in last + prev)
    eligible = enough and sum(x[-1] > 0 for x in last) >= 2 and sum(map(sum, last)) >= sum(map(sum, prev))
    decision = dict(lower_lr_extension_validation_gate=eligible,
        automatic_extension=False, reason="Needs a separately fixed extension protocol" if eligible
        else "Final validation wins or late performance retention did not meet the preregistered gate")
    write_json(out / "continuation_decision.json", decision)
    lines += ["", "## 추가 학습 판단", "",
        ("낮은 학습률 조건은 사전 검증 기준을 충족했다. 추가 학습 예산과 평가 계획을 별도로 고정할 수 있다."
         if eligible else "낮은 학습률 조건은 사전 검증 기준을 충족하지 못했다. 같은 설정의 자동 연장은 하지 않는다."),
        "시험 점수로 모델이나 설정을 다시 선택하지 않았다. 3개 학습 시드는 초기 비교이며 확정적 우열을 뜻하지 않는다.",
        "", "## 보존된 대표 모델", "",
        f"검증으로 사전 선택: {selected['reward']}, seed {selected['seed']}, {selected['step']}스텝.",
        f"별도 시험: {test['summary']['kills']}/{plan['test_n']}승.",
        "대표 모델: selected_policy/policy_net.zip. 마지막 모델과 구분한다.", "",
        "![학습 곡선](learning_curves.png)", "",
        "![고정 검증 경기 리플레이](flight_replay.gif)", "",
        "리플레이는 승리 경기를 골라낸 것이 아니라 고정된 기존 검증 seed 900000의 양 좌석이다.",
        "Q값 진단과 중간 예산 결과는 results.md 및 summary.json에 있다.", ""]
    (out / "결과.md").write_text("\n".join(lines), encoding="utf-8")
    files = [p for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in (".json", ".zip", ".pkl", ".png", ".gif", ".md")
        and p.name not in ("artifact_manifest.json", "completion.json")]
    write_json(out / "artifact_manifest.json", [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    write_json(out / "completion.json", dict(status="complete", finished_utc=timestamp(),
        training_steps=2 * len(plan["seeds"]) * plan["budgets"][-1], decision=decision))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--plot-python", required=True)
    p.add_argument("--wait-pid", type=int)
    args = p.parse_args()
    out = args.out.resolve()
    try:
        write_json(out / "postprocess_status.json", dict(status="waiting" if args.wait_pid else "running",
            training_pid=args.wait_pid, updated_utc=timestamp()))
        if args.wait_pid:
            # Open a process handle once to avoid confusing a reused process ID.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel.WaitForSingleObject.restype = ctypes.c_uint32
            kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel.OpenProcess(0x00100000, False, args.wait_pid)
            if not handle and not (out / "timing.json").exists():
                raise RuntimeError("Cannot monitor the specified training process.")
            try:
                while handle and kernel.WaitForSingleObject(handle, 0) == 258:
                    time.sleep(10)
            finally:
                if handle:
                    kernel.CloseHandle(handle)
        if not (out / "timing.json").exists():
            raise RuntimeError("Training/evaluation did not finish; inspect worker logs.")
        write_json(out / "postprocess_status.json", dict(status="running", updated_utc=timestamp()))
        finalize(out, args.plot_python)
        write_json(out / "postprocess_status.json", dict(status="complete", updated_utc=timestamp()))
    except Exception as exc:
        write_json(out / "completion.json", dict(status="failed", error=str(exc), updated_utc=timestamp()))
        raise


if __name__ == "__main__":
    main()
