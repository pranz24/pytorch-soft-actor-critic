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

#### For SAC :

```
python main.py --env-name Humanoid-v2 --scale_R 20 
```

#### For SAC (Hard Update):

```
python main.py --env-name Humanoid-v2 --scale_R 20 --tau 1 --target_update_interval 1000
```

#### For SAC (Deterministic, Hard Update):

```
python main.py --env-name Humanoid-v2 --scale_R 20 --deterministic True --tau 1 --target_update_interval 1000
```

### Results
------------
My results on Humanoid-v2 environment using SAC, SAC(hard update) and SAC(deterministic, hard update).
This is a plot of average rewards at every 10000 step interval 

![sac all](https://user-images.githubusercontent.com/18737539/45465027-f5813900-b730-11e8-8a5d-37a550e1971f.jpeg)

### Hyperparameters
-------------

Use the following hyperparameters for different environment:

| Parameters     | Value  |
| --------------- | ------------- |
|**Shared**|-|
| optimizer | Adam |
| learning rate  | 3x10<sup>−4</sup> |
| discount (γ) | 0.99 |
| replay buffer size | 1x10<sup>6</sup> |
|number of hidden layers (all networks)|2|
|number of hidden units per layer|256|
|number of samples per minibatch|256|
|nonlinearity|ReLU|
|**SAC**|-|
|target smoothing coefficient (τ)|0.005|
|target update interval|1|
|gradient steps|1|
|**SAC** *(Hard Update)*|-|
|target smoothing coefficient (τ)|1|
|target update interval|1000|
|gradient steps (except humanoids)|4|
|gradient steps (humanoids)|1|




| Environment     | Reward Scale  |
| --------------- | ------------- |
| HalfCheetah-v2  | 5 |
| Hopper-v2       | 5 |
| Walker2d-v2     | 5 |
| Ant-v2          | 5 |
| Humanoid-v2     | 20 |
