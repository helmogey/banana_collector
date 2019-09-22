# banana_collector

This is the first project in the Udacity Deep Reinforcement Learning Nanodegree.




# The Environment


This project will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right <br />
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.



To achieve this I used a deep Q-network (DQN) which is able to combine reinforcement learning with a class of artificial neural network known as deep neural networks in which several layers of nodes are used to build up progressively more abstract representations of the data and have made it possible for artificial neural networks to learn concepts such as object categories directly from raw sensory data.


## Getting Started
See the instrucions here to set up your environment [instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies) 

It also requires [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/) 


Get the environment matching your OS :

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


Use full path file reference for such environment. Note that Banana.app is already included in this repo, so it can be imported with: 
```
env = UnityEnvironment(file_name="Banana.app")
```

## Instructions
Then run the [`navigation_banana.ipynb`](https://github.com/doctorcorral/DRLND-p1-banana/blob/master/navigation_banana.ipynb) notebook using the drlnd kernel to train the DQN agent.

After trainig the model, parameters will be dumpt to `checkpoint.pth` and will be used by the trained agent.

