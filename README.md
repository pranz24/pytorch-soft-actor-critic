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

### Results
------------
My results on Humanoid-v2 environment using SAC, SAC(hard update) and SAC(deterministic, hard update).
This is a plot of average rewards at every 10000 step interval 

![sac all](https://user-images.githubusercontent.com/18737539/45465027-f5813900-b730-11e8-8a5d-37a550e1971f.jpeg)

