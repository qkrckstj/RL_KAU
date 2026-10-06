"""Paired replay-capacity continuation, with all selection before held-out testing."""
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import subprocess
import sys
from tools.plan_a import ROOT, timestamp, write_json
from tools.plan_a_shaping import read, sha
from tools.resume_dqn import resume


def worker(out, capacity):
    p = read(out / "plan.json")
    for name, expected in p["source_hashes"].items():
        if sha(Path(p["source"])/"checkpoints/step_1024000"/name) != expected:
            raise ValueError("Source checkpoint changed")
    for name, expected in p["code_hashes"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("Code changed after protocol was fixed")
    return resume(Namespace(run=p["source"],checkpoint=1024000,out=str(out/f"buffer_{capacity}"),
        additional_steps=p["additional_steps"],seed=10002,device="cpu",val_every=p["val_every"],val_n=p["val_n"],
        replay_capacity=capacity,replay_milestones=[1024000,p["target_steps"]],replay_diagnostics=True))


def child(out, capacity):
    with (out / f"buffer_{capacity}.log").open("w",encoding="utf-8") as log:
        result = subprocess.run([sys.executable,"-X","utf8","-u","-m","tools.compare_replay",
            "--out",str(out),"--worker",str(capacity)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONIOENCODING="utf-8"),
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    if result.returncode:
        raise RuntimeError(f"Buffer {capacity} failed; see its log")


def run(out, smoke):
    if out.exists():
        raise FileExistsError("Use a new experiment folder")
    source = ROOT / "runs/plan_a_20261005/double/ddqn/s1"
    checkpoint = source / "checkpoints/step_1024000"
    paths = ["tools/compare_replay.py","tools/resume_dqn.py","tools/plan_a.py","tools/grade.py",
             "tools/policies.py","tests/test_replay_resize.py"]
    steps = 128 if smoke else 1024000
    p = dict(created_utc=timestamp(),source=str(source),capacities=[50000,500000],
        exploratory=True,additional_steps=steps,target_steps=1024000+steps,environment_seed=10002,
        val_band=900000,val_n=2 if smoke else 20,val_every=128 if smoke else 51200,
        test_band=900000 if smoke else 1700000,test_n=2 if smoke else 40,
        changes="Replay capacity only; both arms repacked oldest-first with all 50000 transitions retained",
        controls="Same saved learner, optimizer, CPU RNG, fresh environment seed, epsilon=.05, lr=.001 and terminal rewards",
        limits="Exploratory paired continuation of one selected training seed; not independent replication",
        selection="Validation wins, mean win time, earlier checkpoint; starting model eligible; no test-driven reselection",
        extension="Fixed endpoint at 2048000; no automatic extension",
        source_hashes={n:sha(checkpoint/n) for n in ("policy_net.zip","replay_buffer.pkl","rng_state.pkl","resume.json")},
        code_hashes={n:sha(ROOT/n) for n in paths})
    out.mkdir(parents=True)
    write_json(out/"plan.json",p)
    for name in paths:
        dest=out/"source_snapshot"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    shutil.copyfile(source/"utils.py",out/"source_snapshot/frozen_utils.py")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda c:child(out,c),p["capacities"]))
        choices=[]
        for capacity in p["capacities"]:
            d=out/f"buffer_{capacity}"
            r=read(d/"result.json")
            for kind,step,weights in (("starting",1024000,"checkpoints/step_1024000/policy_net.zip"),
                ("validation_selected",r["best_step"],"policy_net.zip"),("final",r["steps"],"final_net.zip")):
                choices.append(dict(capacity=capacity,kind=kind,step=step,weights=weights,sha256=sha(d/weights)))
        write_json(out/"selection.json",dict(frozen_utc=timestamp(),choices=choices))
        from tools.grade import score
        from tools import project
        import torch
        torch.set_num_threads(1)
        proj=project.load(str(ROOT/"templates/project_04_fair"))
        rows=[]
        for c in choices:
            d=out/f"buffer_{c['capacity']}"
            if sha(d/c["weights"]) != c["sha256"]:
                raise ValueError("Frozen weights changed")
            test=score(d,d/c["weights"],proj,p["test_band"],p["test_n"])
            rows.append(dict(**c,test=test))
            write_json(out/"test_progress.json",rows)
        write_json(out/"summary.json",dict(rows=rows,finished_utc=timestamp()))
        lines=["# 재생 버퍼 5만 / 50만 비교", "",
            "같은 Double DQN 시드 1 체크포인트에서 이어 학습한 탐색적 비교다. 시작/검증 최고/마지막 모델을 같은 새 시험에서 평가했다.", "",
            "| 버퍼 크기 | 모델 | 스텝 | 시험 승리 |","|---:|---|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['capacity']} | {r['kind']} | {r['step']} | {r['test']['summary']['kills']}/{p['test_n']} |")
        lines += ["","검증 이력의 replay_diagnostics에 양의 보상 경험 수·Q값 이탈·버퍼 크기를 기록했다.",
            "학습 재개용 replay/RNG는 시작과 마지막 체크포인트에 보관했다. 중간에는 가중치만 저장했다."]
        (out/"결과.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        for name,expected in p["source_hashes"].items():
            if sha(checkpoint/name)!=expected:
                raise ValueError("Original checkpoint changed")
        write_json(out/"completion.json",dict(status="complete",finished_utc=timestamp()))
    except Exception as exc:
        write_json(out/"completion.json",dict(status="failed",error=str(exc),updated_utc=timestamp()))
        raise


if __name__ == "__main__":
    ap=ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--worker",type=int,choices=[50000,500000])
    args=ap.parse_args()
    if args.worker:
        worker(args.out.resolve(),args.worker)
    else:
        run(args.out.resolve(),args.smoke)
