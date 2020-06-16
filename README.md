
# Multiagent PG/A2C with safety constraint (AdaTD)

Add safety constraint to vanilla multiagent policy gradient (PG) and advantage actor-critic (A2C). The safety constraint is ensured by lagrange relaxation during training.

## Getting started:

- Dependencies: Python (>=3.7), OpenAI gym (0.10.5), tensorflow (>=2.0.0), multiagent, matplolib

- To install dpendency 'multiagent': Download 'multiagent-particle-envs-master' provided in master branch. Then cd into its root directory and install it using 'pip install -e multiagent-particle-envs-master'

## Code structure

- `train_sma2c.py`: Code for running multiagent advantage actor-critic with safety constraint.

- `optimzers_nonlinear.py`: Code for running multiagent policy gradient with safety constraint.

## Example result
An example reward plot of safe multiagent A2C method.
![alt text](https://github.com/hanshen95/multiagent-pg-a2c-with-safety-constraint/blob/master/Mean_Episode_Rewards_SMA2C.png?raw=true)

An example safety cost plot of safe multiagent A2C method. The cost is supposed to be smaller than a safety threshold after convergence. In this example plot, cost goes to 0.
![alt text](https://github.com/hanshen95/multiagent-pg-a2c-with-safety-constraint/blob/master/Mean_Episode_Costs_SMA2C.png?raw=true)

## Acknowlegement

The dependency 'multiagent' used in this repo is a modified version of [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs/blob/master/README.md#multi-agent-particle-environment)
