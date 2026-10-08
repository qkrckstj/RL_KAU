"""Self-contained contextual-speed export; does not launch any experiment."""
from pathlib import Path
import json
import zipfile
from experiments.league.contextual_speed import Policy
from tools.autolab_cem import ROOT,sha
from tools.league_reliable_io import write
from tools.league_separated_export import export as export_base


def export(out,configuration,identity):
    Policy(configuration=configuration)
    out=Path(out)
    spec=export_base(out,{k:configuration[k] for k in ('probe','parent','interceptor')},identity)
    extension=(ROOT/'experiments/league/contextual_speed.py').read_text('utf-8')
    extension=extension.replace('from experiments.league.separated_probe import Policy as Separated','Separated = Policy')
    extension=extension.replace('from experiments.league.controller import Policy as Expert','')
    (out/'policy.py').write_text((out/'policy.py').read_text('utf-8')+'\n'+extension,'utf-8')
    write(out/'policy_net.json',configuration)
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('parameters.json',json.dumps(configuration))
    write(out/'artifact_sha256.json',{n:sha(out/n) for n in ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec
