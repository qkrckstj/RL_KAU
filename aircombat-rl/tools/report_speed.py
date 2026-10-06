"""Wait for the heading-only experiment, summarize, and show a local notification."""
from argparse import ArgumentParser
from pathlib import Path
import os
import statistics
import subprocess
import time
from tools.plan_a_shaping import read
from tools.plan_a import write_json,timestamp


def run(out):
    write_json(out/"notification_status.json",dict(stage="waiting",updated_utc=timestamp()))
    deadline=time.monotonic()+6*3600
    while not (out/"completion.json").exists():
        if time.monotonic()>deadline:
            raise TimeoutError("Experiment did not complete in six hours")
        time.sleep(10)
    result=read(out/"completion.json")
    if result["status"]=="complete":
        summary=read(out/"summary.json")
        lines=["감속 고정·선회 학습 및 평가가 완료됐습니다.",""]
        for condition,label in (("heading_only","새 감속 고정 모델"),("ddqn_reference","기존 Double DQN")):
            for kind,title in (("validation_selected","검증 최고"),("budget_endpoint","마지막")):
                rows=[r for r in summary["rows"] if r["reward"]==condition and r["kind"]==kind]
                wins=[r["test"]["summary"]["kills"] for r in rows]
                rates=[r["test"]["summary"]["kills"]/r["test"]["summary"]["n"] for r in rows]
                lines.append(f"{label} {title}: 시드별 {wins}, 평균 {statistics.mean(rates):.1%}")
        b=summary["fixed_baseline"]["summary"]
        lines += [f"고정 좌선회·감속: {b['kills']}/{b['n']}승.","","시험 점수로 모델을 재선택하지 않았습니다.","추가 학습은 예약돼 있지 않습니다."]
    else:
        lines=["학습 또는 평가가 실패했습니다.",str(result.get("error")),"아래 폴더의 로그를 확인하세요."]
    lines += ["",str(out)]
    report=out/"완료보고.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    command=("Add-Type -AssemblyName System.Windows.Forms; "
        "$report = Get-Content -LiteralPath $env:RL_REPORT_PATH -Raw -Encoding UTF8; "
        "[System.Windows.Forms.MessageBox]::Show($report, 'RL experiment complete') | Out-Null")
    subprocess.Popen(["powershell.exe","-NoProfile","-Command",command],env=dict(os.environ,RL_REPORT_PATH=str(report)),
        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    write_json(out/"notification_status.json",dict(stage="notified",updated_utc=timestamp()))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    run(p.parse_args().out.resolve())
