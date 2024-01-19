# ACH: Adaptive Communication-Based Multi-Agent Reinforcement Learning for Heterogeneous Agent Cooperation

Code for **Adaptive Communication-Based Multi-Agent Reinforcement Learning for Heterogeneous Agent Cooperation**. ACH is implemented in PyTorch and tested on the SPP tasks, which is based on [PyMARL](https://github.com/oxwhirl/pymarl).


## Python MARL framework

This PyMARL includes baselines of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

## Installation instructions
### Requirements
- Python 3.6+
- pip packages listed in requirements.txt

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recommended).

## Run an experiment 

```shell
python src/main.py
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.


