"""Render recorded diagnostic flight traces with no learner dependency."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(out):
    names=("starting_good","continued_bad","large_buffer_bad","small_buffer_recovered")
    fig,axes=plt.subplots(4,4,figsize=(16,12))
    for col,name in enumerate(names):
        records=json.loads((out/f"{name}_hold1/traces.json").read_text(encoding="utf-8"))
        r=next(x for x in records if x["seed"]==900000 and x["seat"]=="red")
        f=r["frames"]
        t=[x["t"] for x in f]
        ax=axes[0,col]
        for prefix,color in (("own","#2864a0"),("opp","#c94343")):
            ax.plot([x[f"{prefix}_x"]/1000 for x in f],[x[f"{prefix}_y"]/1000 for x in f],color=color,label=prefix)
        ax.set(title=f"{name}\n{r['outcome']}",xlabel="East km",ylabel="North km",aspect="equal")
        ax.legend(fontsize=8)
        axes[1,col].step(t,[x["dh"] for x in f],where="post")
        axes[1,col].set(ylabel="Heading action (deg)",ylim=(-35,35))
        axes[2,col].plot(t,[x["speed"] for x in f])
        axes[2,col].set(ylabel="Speed (kt)")
        for key,color in (("own_health","#2864a0"),("opp_health","#c94343")):
            axes[3,col].plot(t,[x[key] for x in f],color=color,label=key)
        axes[3,col].set(ylabel="Health",xlabel="Time (s)",ylim=(-.02,1.02))
        for row in range(4):
            axes[row,col].grid(alpha=.2)
    fig.suptitle("Fixed validation seed 900000 / red seat | traces sampled every 0.2s; reversal metrics use every 0.05s")
    fig.tight_layout()
    fig.savefig(out/"flight_comparison.png",dpi=130)
    plt.close(fig)
    rows=json.loads((out/"summary.json").read_text(encoding="utf-8"))["rows"]
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    for hold,offset,color in ((1,-.18,"#2864a0"),(4,.18,"#dc7428")):
        selected=[next(r for r in rows if r["model"]==n and r["hold"]==hold) for n in names]
        for ax,key,title in zip(axes,("wins","heading_reversals_per_second","wez_time"),
            ("Wins / 20 validation matches","Heading reversals / second","Own weapon-zone time (s)")):
            ax.bar([i+offset for i in range(4)],[r[key] for r in selected],width=.35,color=color,label=f"Hold {hold}/20 s")
            ax.set(title=title,xticks=range(4),xticklabels=["Start","Continued","Large buffer","Small buffer"])
            ax.tick_params(axis="x",rotation=20)
            ax.grid(axis="y",alpha=.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out/"diagnostic_metrics.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    run(p.parse_args().out)
