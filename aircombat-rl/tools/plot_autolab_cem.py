"""Render completed CEM results with matplotlib, independent of the RL runtime."""
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/autolab_20261005"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(warm=False,followup=None,control_directory=None):
    prefix="cem_warm" if warm else "cem"
    followup=followup or BASE/f"{prefix}_followup"
    out=followup/"report"
    out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,folder,title in ((axes[0,0],BASE/f"{prefix}_pilot","Development screening"),
                            (axes[0,1],followup/"replication","Independent replication")):
        summary=read(folder/"summary.json")
        plan=read(folder/"plan.json")
        for record in summary["records"]:
            seed=record["seed"]
            history=read(folder/f"s{seed}/validation.json")
            ax.plot([r["generation"]+1 for r in history],[100*r["result"]["summary"]["rate"] for r in history],
                    marker="o",label=f"Search seed {seed}")
        ax.axhline(100*summary["fixed_baseline"]["summary"]["rate"],color="black",linestyle="--",label="Fixed left + decelerate")
        ax.set(title=f"{title}: {plan['validation_n']} matches",xlabel="Search generation (0 = parent)",ylabel="Win rate (%)",
               ylim=(0,105),xticks=range(0 if warm else 1,plan['generations']+1))
        ax.legend(fontsize=8)
    verdict=read(followup/"heldout/verdict.json")
    seeds=read(followup/"replication/plan.json")["seeds"]
    ax=axes[1,0]
    x=np.arange(3)
    for offset,values,label in ((-.18,verdict["selected_rates"],"Development-selected"),(.18,verdict["final_rates"],"Final generation")):
        bars=ax.bar(x+offset,np.array(values)*100,width=.35,label=label)
        ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
    ax.axhline(100*verdict["baseline"],color="black",linestyle="--",label="Fixed left + decelerate")
    ax.set(title="Sealed evaluation: 200 matches per policy",ylabel="Win rate (%)",xticks=x,xticklabels=[f"Seed {s}" for s in seeds],ylim=(0,110))
    ax.legend(fontsize=8)
    controls=read((control_directory or BASE/f"{prefix}_controls")/"summary.json")
    values=[verdict["baseline"],controls["untuned_initial"]["rate"],controls["untuned_anchor"]["rate"]]
    labels=["Fixed action 0","Untuned\ninitial","Untuned\nanchor"]
    if "learned_parent" in controls:
        values.append(controls["learned_parent"]["rate"])
        labels.append("Learned\nparent")
    values.append(verdict["selected_mean"])
    labels.append("Refinement\n3-run mean" if warm else "Learned\n3-run mean")
    ax=axes[1,1]
    bars=ax.bar(range(len(values)),np.array(values)*100,color=["#aaaaaa"]*(len(values)-1)+["#227f62"])
    ax.bar_label(bars,fmt="%.1f",padding=3)
    ax.set(title="Sealed evaluation: learned vs controls",ylabel="Win rate (%)",xticks=range(len(values)),
           xticklabels=labels,ylim=(0,110))
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    lo,hi=verdict["paired_ic_bootstrap_95_ci"]
    fig.suptitle("FairFight vs unchanged Ace | " + ("CEM refinement of one shared learned parent" if warm else "CEM parameter learning"))
    fig.text(.5,.013,f"Held-out: 100 shared initial conditions x 2 seats. Advantage over fixed action 0: {100*verdict['advantage']:.1f} pp; paired IC bootstrap 95% CI [{100*lo:.1f}, {100*hi:.1f}] pp.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    fig.savefig(out/"learning_and_test.png",dpi=160)
    plt.close(fig)
    trace_file=out/"flight/traces.json"
    if not trace_file.exists():
        return
    traces=read(trace_file)
    for seat in ("red","blue"):
        fig,axes=plt.subplots(4,2,figsize=(11,12))
        pair=[r for r in traces if r["seat"]==seat]
        all_frames=[frame for r in pair for frame in r["frames"]]
        all_x=[frame[p+"_x"]/1000 for frame in all_frames for p in ("own","opp")]
        all_y=[frame[p+"_y"]/1000 for frame in all_frames for p in ("own","opp")]
        x_limits=(min(all_x)-.3,max(all_x)+.3)
        y_limits=(min(all_y)-.3,max(all_y)+.3)
        end_time=max(frame["t"] for frame in all_frames)
        for col,name in enumerate(("fixed_left","learned")):
            r=next(r for r in traces if r["seat"]==seat and r["model"]==name)
            f=r["frames"]
            t=[a["t"] for a in f]
            for prefix,color in (("own","#2864a0"),("opp","#c94343")):
                axes[0,col].plot([a[prefix+"_x"]/1000 for a in f],[a[prefix+"_y"]/1000 for a in f],color=color,label=prefix)
                axes[3,col].plot(t,[a[prefix+"_health"] for a in f],color=color,label=prefix)
            axes[0,col].set(title=f"{name}: {r['outcome']}",xlabel="East (km)",ylabel="North (km)",aspect="equal",xlim=x_limits,ylim=y_limits)
            axes[0,col].legend()
            axes[1,col].step(t,[a["dh"] for a in f],where="post")
            axes[1,col].set(ylabel="Heading command (deg)",ylim=(-35,35))
            axes[2,col].plot(t,[a["speed"] for a in f])
            axes[2,col].set(ylabel="Speed (kt)")
            axes[3,col].set(ylabel="Health",xlabel="Time (s)",ylim=(-.02,1.02))
            for row in range(4):
                axes[row,col].grid(alpha=.2)
                if row>0:
                    axes[row,col].set_xlim(0,end_time)
        fig.suptitle(f"Prechosen development seed 904000 / {seat} seat | sample flight, not an aggregate score")
        fig.tight_layout()
        fig.savefig(out/f"flight_{seat}.png",dpi=140)
        plt.close(fig)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--warm",action="store_true")
    parser.add_argument("--followup",type=Path)
    parser.add_argument("--controls",type=Path)
    args=parser.parse_args()
    run(args.warm,args.followup,args.controls)
