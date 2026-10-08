"""Load an experimental GPU-trained policy for the original JSBSim evaluator."""
import torch
from experiments.plan_a.core import State,specification
from experiments.gpu_sim.train import Agent


class Policy:
    ACTION_MODE='discrete'

    def __init__(self,weights=None,device='cpu'):
        if weights is None: raise ValueError('A GPU-training policy.pt checkpoint is required')
        self.device=torch.device(device)
        saved=torch.load(weights,map_location=self.device,weights_only=True)
        self.model=Agent().to(self.device)
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        self.state=State(specification('observed_ppo'))

    @torch.no_grad()
    def act(self,obs):
        x=torch.as_tensor(self.state(obs),device=self.device)
        logits,_=self.model(x)
        return int(logits.argmax())
