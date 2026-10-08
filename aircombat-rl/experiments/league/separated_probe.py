"""Fixed .5-second probe and interceptor, with a separate post-probe parent.

Only the same 39 public channels enter the unchanged direction-invariant gate.
Separating the initial actions permits fitting the parent without discarding
the already learned opening. No opponent labels or simulator internals enter.
"""
import json
import zipfile
from experiments.league.adaptive_symmetric import Policy as Gate
from experiments.league.controller import Policy as Expert
from experiments.league.interception import Policy as Interceptor


class ParentAfterProbe:
    def __init__(self, owner, probe, parent):
        self.owner, self.probe, self.parent = owner, Expert(parameters=probe), Expert(parameters=parent)

    def act(self, obs):
        return (self.probe if self.owner.selected is None else self.parent).act(obs)


class Policy(Gate):
    def __init__(self, weights=None, device='cpu', configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as archive:
                configuration = json.loads(archive.read('parameters.json'))
        super().__init__(configuration=dict(experts=[dict(parameters=configuration['probe'])]*2,
            gate=[.5, 1., -8., 0., -8., 0.]))
        self.experts[0] = ParentAfterProbe(self, configuration['probe'], configuration['parent'])
        self.experts[1] = Interceptor(parameters=configuration['interceptor'])

    def __str__(self):
        return 'Fixed public-motion gate with separated probe and post-probe parent'
