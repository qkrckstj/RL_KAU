import pytest
import torch


def test_gae_stops_at_real_episode_end():
    from experiments.gpu_sim.train import advantages
    rewards=torch.tensor([[1.],[2.]])
    values=torch.tensor([[.5],[.6]])
    done=torch.tensor([[True],[False]])
    adv,returns=advantages(rewards,values,done,torch.tensor([3.]),gamma=.9,lam=1.)
    torch.testing.assert_close(adv,torch.tensor([[.5],[4.1]]))
    torch.testing.assert_close(returns,torch.tensor([[1.],[4.7]]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_native_gpu_ppo_updates_and_checkpoint_reloads(tmp_path):
    pytest.importorskip('jsbsim_f16_cuda')
    from experiments.gpu_sim.train import main,Agent
    out=tmp_path/'training'
    main(['--out',str(out),'--envs','8','--rollout','8','--iterations','2','--epochs','1','--batch-size','64'])
    saved=torch.load(out/'policy.pt',weights_only=True)
    assert saved['metrics']['optimizer_steps'] == 2
    assert saved['metrics']['weights_changed']
    assert saved['metrics']['invalid_episodes'] == 0
    model=Agent().cuda()
    model.load_state_dict(saved['model'])
    logits,values=model(torch.zeros(8,31,device='cuda'))
    assert logits.shape == (8,9) and torch.isfinite(values).all()
    from experiments.gpu_sim.policy import Policy
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    official=FairFightEnv()
    try:
        raw,_=official.reset(seed=33030000)
        cpu_policy=Policy(out/'policy.pt')
        assert 0 <= cpu_policy.act(raw) < 9
    finally:
        official.close()
    with pytest.raises(FileExistsError):
        main(['--out',str(out)])
