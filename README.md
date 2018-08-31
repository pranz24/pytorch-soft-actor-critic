### Description
------------
Reimplementation of [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

Contributions are welcome. If you find any mistake (very highly possible) or know how to make it more stable, don't hesitate to send a pull request.

### Requirements
------------

- [mujoco-py](https://github.com/openai/mujoco-py)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)

### Run
------------
Use the default hyperparameters.

#### For SAC (Gaussian Policy):

```
python main.py --algo SAC --env-name HalfCheetah-v2
```
#### For SAC (Gaussian Mixture Policy):

```
python main.py --algo SAC(GMM) --env-name HalfCheetah-v2 --k 4
```

### TODO
------------
- [x] Gaussian Policy
- [x] Reparameterization
- [x] Gaussian Mixture Model
- [ ] Deterministic policy
- [ ] Soft Actor-Critic (hard target update)
- [ ] Evaluate the trained Policy

