"""Cross-evaluate frozen CEM candidates on the two already-open development sets."""
from concurrent.futures import ProcessPoolExecutor
from tools.autolab_cem import ROOT,read,write,sha
from tools.autolab_cem_followup import holdout_job


def main():
    base=ROOT/"runs/autolab_20261005"
    out=base/"cem_crosscheck"
    if out.exists():
        raise FileExistsError("Crosscheck directory already exists")
    out.mkdir()
    jobs=[]
    for source,seeds,band,n in ((base/"cem_pilot",[800,801,802],904000,40),
                               (base/"cem_followup/replication",[803,804,805],903000,20)):
        for seed in seeds:
            d=source/f"s{seed}"
            jobs.append(dict(name=f"s{seed}",parameters=read(d/"policy_net.json")["parameters"],band=band,n=n,
                output=str(out/f"s{seed}.json"),source=str(d),weights_sha256=sha(d/"policy_net.json")))
    write(out/"plan.json",dict(jobs=jobs,purpose="Development crosscheck only; no heldout data"))
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=dict(pool.map(holdout_job,jobs))
    write(out/"summary.json",{name:result["summary"] for name,result in results.items()})
    write(out/"completion.json",dict(status="complete"))


if __name__=="__main__":
    main()
