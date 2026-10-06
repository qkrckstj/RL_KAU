"""Fixed-budget exploratory continuation of the late-improving DDQN seed 1."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import torch
from tools.plan_a import ROOT, timestamp, write_json
from tools.plan_a_shaping import read, sha
from tools.resume_dqn import resume
from tools.grade import score
from tools import project


def run(out, smoke=False):
    if out.exists():
        raise FileExistsError("Use a new experiment directory")
    source = ROOT / "runs/plan_a_20261005/double/ddqn/s1"
    checkpoint = source / "checkpoints/step_1024000"
    config = read(source / "config.json")
    if not config["spec"].get("double_dqn"):
        raise ValueError("Expected Double DQN")
    out.mkdir(parents=True)
    steps = 128 if smoke else 1024000
    plan = dict(created_utc=timestamp(), exploratory=True, source=str(source), checkpoint=1024000,
        additional_steps=steps, target_steps=1024000+steps, environment_seed=10002,
        val_band=900000, val_n=2 if smoke else 20, val_every=128 if smoke else 51200,
        test_band=900000 if smoke else 1600000, test_n=2 if smoke else 40,
        selection="Validation wins, mean kill time, earlier checkpoint; starting policy remains eligible",
        limits="One previously inspected seed; not independent confirmation. Fresh simulator episode, restored learner/replay/CPU RNG.",
        source_hashes={name:sha(checkpoint/name) for name in ("policy_net.zip","replay_buffer.pkl","rng_state.pkl","resume.json")})
    write_json(out / "plan.json", plan)
    snap = out / "source_snapshot"
    snap.mkdir()
    for path in (Path(__file__), ROOT / "tools/resume_dqn.py", ROOT / "tools/plan_a.py", source / "utils.py"):
        shutil.copyfile(path, snap/path.name)
    dest = out / "learner"
    try:
        result = resume(Namespace(run=str(source), checkpoint=1024000, out=str(dest),
            additional_steps=steps, seed=10002, device="cpu", val_every=plan["val_every"], val_n=plan["val_n"]))
        choices = [dict(kind=kind, step=step, weights=weights, sha256=sha(dest/weights))
            for kind,step,weights in (
                ("starting_policy",1024000,"checkpoints/step_1024000/policy_net.zip"),
                ("validation_selected",result["best_step"],"policy_net.zip"),
                ("final_policy",result["steps"],"final_net.zip"))]
        write_json(out / "selection.json", dict(frozen_utc=timestamp(), choices=choices))
        torch.set_num_threads(1)
        proj = project.load(str(ROOT / "templates/project_04_fair"))
        cache, rows = {}, []
        for c in choices:
            if sha(dest/c["weights"]) != c["sha256"]:
                raise ValueError("Checkpoint changed after selection")
            if c["sha256"] not in cache:
                cache[c["sha256"]] = score(dest,dest/c["weights"],proj,plan["test_band"],plan["test_n"])
            rows.append(dict(**c,test=cache[c["sha256"]]))
        write_json(out / "summary.json", dict(result=result, rows=rows))
        lines = ["# Double DQN 시드 1 추가 학습", "",
            "시작 정책과 검증 최고·마지막 정책을 동일한 새 시험 조건에서 비교했다.",
            "이미 관찰한 단일 시드의 탐색적 후속 실험이며, 독립적인 성능 확인 실험은 아니다.", "",
            "| 모델 | 스텝 | 시험 승리 |", "|---|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['kind']} | {r['step']} | {r['test']['summary']['kills']}/{plan['test_n']} |")
        (out / "결과.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        for name, expected in plan["source_hashes"].items():
            if sha(checkpoint/name) != expected:
                raise ValueError("Original checkpoint changed")
        write_json(out / "completion.json", dict(status="complete", finished_utc=timestamp()))
    except Exception as exc:
        write_json(out / "completion.json", dict(status="failed", error=str(exc), updated_utc=timestamp()))
        raise


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.out.resolve(), args.smoke)
