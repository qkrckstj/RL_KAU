"""Frozen symmetric gate and parent expert, with a trainable interceptor."""
import json
import zipfile
from experiments.league.adaptive_symmetric import Policy as Gate
from experiments.league.interception import Policy as Interceptor


class Policy(Gate):
    def __init__(self, weights=None, device='cpu', configuration=None):
        if configuration is None:
            with zipfile.ZipFile(weights) as archive:
                configuration = json.loads(archive.read('parameters.json'))
        parent = configuration['parent']
        super().__init__(configuration=dict(experts=[dict(parameters=parent)]*2,
            gate=[.5, 1., -8., 0., -8., 0.]))
        self.experts[1] = Interceptor(parameters=configuration['interceptor'])

    def __str__(self):
        return 'Fixed public-motion gate with seven-parameter interceptor'
