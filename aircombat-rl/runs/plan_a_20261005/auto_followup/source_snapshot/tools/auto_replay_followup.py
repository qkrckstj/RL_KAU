"""Finish replay comparison, optionally run one fixed lower-rate continuation, report.

The branch uses validation only, never the parent's held-out test scores.
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import torch
from tools.plan_a import ROOT, timestamp, write_json
from tools.plan_a_shaping import read, sha
from tools.plan_a_long import select_at_budget
from tools.resume_gentle import resume


def retention(history):
    start = history[0]["summary"]["kills"]
    end = history[-1]["summary"]["kills"]
    recent = [h["summary"]["kills"] for h in history[-5:]]
    good = start > 0 and end >= start and sum(recent)/len(recent) >= start/2
    return dict(start=start,end=end,recent=recent,retained=good)


def notify(out):
    # Local completion notice, not a claim of a proactive message in the chat.
    command=("Add-Type -AssemblyName System.Windows.Forms; "
        "$report = Get-Content -LiteralPath $env:RL_REPORT_PATH -Raw -Encoding UTF8; "
        "[System.Windows.Forms.MessageBox]::Show($report, 'RL experiment complete') | Out-Null")
    subprocess.Popen(["powershell.exe","-NoProfile","-Command",command],
        env=dict(os.environ,RL_REPORT_PATH=str(out/"report.md")),
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def run(out,parent,smoke=False,notification=False):
    if out.exists():
        raise FileExistsError("Use a new output directory")
    out.mkdir(parents=True)
    steps=128 if smoke else 512000
    protocol=dict(created_utc=timestamp(),parent=str(parent),parent_plan_sha256=sha(parent/"plan.json"),
        decision="If neither buffer ends at least as good as the starting validation wins AND retains half that win count on average over the last five validations, run the follow-up",
        branch_data="Validation only; parent test scores not used",additional_steps=steps,
        learning_rate=.0001,replay_capacity=50000,environment_seed=10002,
        val_every=128 if smoke else 51200,val_n=2 if smoke else 20,
        test_band=900000 if smoke else 1800000,test_n=2 if smoke else 40,
        budget="One follow-up maximum; no recursive optimization or further extension")
    write_json(out/"plan.json",protocol)
    for name in ("tools/auto_replay_followup.py","tools/resume_gentle.py"):
        dest=out/"source_snapshot"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    write_json(out/"status.json",dict(stage="waiting_for_replay_comparison",updated_utc=timestamp()))
    try:
        deadline=time.monotonic()+4*3600
        while not (parent/"completion.json").exists():
            if time.monotonic()>deadline:
                raise TimeoutError("Replay comparison did not finish within four hours; inspect its logs")
            time.sleep(10)
        if read(parent/"completion.json")["status"]!="complete":
            raise RuntimeError("Parent experiment failed; no follow-up started")
        if sha(parent/"plan.json")!=protocol["parent_plan_sha256"]:
            raise ValueError("Parent protocol changed")
        decisions={str(c):retention(read(parent/f"buffer_{c}/validation.json")) for c in (50000,500000)}
        proceed=not any(d["retained"] for d in decisions.values())
        write_json(out/"decision.json",dict(followup=proceed,validation=decisions,decided_utc=timestamp()))
        parent_rows=read(parent/"summary.json")["rows"]
        lines=["Replay comparison and automatic follow-up", "", "Replay test results (same test set within this stage):"]
        for r in parent_rows:
            lines.append(f"Buffer {r['capacity']}, {r['kind']}: {r['test']['summary']['kills']}/{r['test']['summary']['n']}")
        if proceed:
            write_json(out/"status.json",dict(stage="training_lower_rate",updated_utc=timestamp()))
            source=ROOT/"runs/plan_a_20261005/double/ddqn/s1"
            dest=out/"learner"
            result=resume(Namespace(run=str(source),checkpoint=1024000,out=str(dest),additional_steps=steps,
                seed=10002,device="cpu",val_every=protocol["val_every"],val_n=protocol["val_n"],
                replay_capacity=50000,replay_milestones=[1024000,1024000+steps],
                replay_diagnostics=True,learning_rate=.0001))
            # Match the baseline at exactly the same continuation budget.
            baseline=parent/"buffer_50000"
            bh=read(baseline/"validation.json")
            best=select_at_budget(bh,1024000+steps)
            choices=[]
            for condition,design,kind,step,weights in (
                ("starting",dest,"starting",1024000,"checkpoints/step_1024000/policy_net.zip"),
                ("lr_0.001",baseline,"validation_selected",best["step"],f"checkpoints/step_{best['step']}/policy_net.zip"),
                ("lr_0.001",baseline,"final",1024000+steps,f"checkpoints/step_{1024000+steps}/policy_net.zip"),
                ("lr_0.0001",dest,"validation_selected",result["best_step"],"policy_net.zip"),
                ("lr_0.0001",dest,"final",result["steps"],"final_net.zip")):
                choices.append(dict(condition=condition,design=str(design),kind=kind,step=step,
                    weights=weights,sha256=sha(design/weights)))
            write_json(out/"selection.json",dict(frozen_utc=timestamp(),choices=choices))
            write_json(out/"status.json",dict(stage="evaluating",updated_utc=timestamp()))
            from tools.grade import score
            from tools import project
            proj=project.load(str(ROOT/"templates/project_04_fair"))
            torch.set_num_threads(1)
            rows=[]
            for c in choices:
                design=Path(c["design"])
                if sha(design/c["weights"])!=c["sha256"]:
                    raise ValueError("Frozen weights changed")
                rows.append(dict(**c,test=score(design,design/c["weights"],proj,protocol["test_band"],protocol["test_n"])))
                write_json(out/"test_progress.json",rows)
            write_json(out/"summary.json",dict(rows=rows,result=result))
            lines += ["", "Lower-rate continuation: matched additional budget, new test set:"]
            for r in rows:
                lines.append(f"{r['condition']}, {r['kind']}: {r['test']['summary']['kills']}/{protocol['test_n']}")
        else:
            lines += ["", "Validation retention criterion passed; no extra training started."]
        lines += ["", "One selected seed: exploratory results, not independent replication.","No further training is scheduled.",str(out)]
        (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        write_json(out/"completion.json",dict(status="complete",followup_ran=proceed,finished_utc=timestamp()))
        write_json(out/"status.json",dict(stage="complete",updated_utc=timestamp()))
        if notification:
            notify(out)
    except Exception as exc:
        write_json(out/"completion.json",dict(status="failed",error=str(exc),updated_utc=timestamp()))
        (out/"report.md").write_text(f"Automatic follow-up failed: {exc}\nInspect logs in {out}\n",encoding="utf-8")
        if notification:
            notify(out)
        raise


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--parent",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--notify",action="store_true")
    a=p.parse_args()
    run(a.out.resolve(),a.parent.resolve(),a.smoke,a.notify)
