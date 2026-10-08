"""Original JSBSim policy interface for a native GPU DQN checkpoint."""
import torch
from experiments.plan_a.core import State, specification
from experiments.gpu_sim.dqn.train import QNetwork


class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu'):
        if weights is None:
            raise ValueError('A native GPU DQN policy.pt is required')
        self.device = torch.device(device)
        saved = torch.load(weights, map_location=self.device, weights_only=True)
        if saved['config']['algorithm'] != 'dqn':
            raise ValueError('Expected a DQN checkpoint')
        self.model = QNetwork().to(self.device)
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        self.state = State(specification('observed_dqn'))

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(self.state(obs), device=self.device)
        return int(self.model(x).argmax())
