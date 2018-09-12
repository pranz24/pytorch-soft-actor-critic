### Description
------------
Reimplementation of [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

Contributions are welcome. If you find any mistake (very likely) or know how to make it more stable, don't hesitate to send a pull request.

### Requirements
------------

- [mujoco-py](https://github.com/openai/mujoco-py)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)

### Run
------------
Use the default hyperparameters.

#### For SAC :

```
python main.py --env-name Humanoid-v2 --scale_R 20 
```

#### For SAC (Hard Update):

```
python main.py --env-name Humanoid-v2 --scale_R 20 --tau 1 --value_update 1000
```

#### For SAC (Deterministic, Hard Update):

```
python main.py --env-name Humanoid-v2 --scale_R 20 --deterministic True --tau 1 --value_update 1000
```

### TODO
------------

- [x] Gaussian Policy
- [x] Reparameterization
- [x] Gaussian Mixture Model
- [x] Use 2 Q-functions
- [x] Evaluate the trained Policy
- [ ] Deterministic Policy (hard target update)

