
# Report
---


## Learning algorithm

The agent uses `dqn` function in the navigation notebook. 

It continues episodical training via a dqn agent until `n_episodes` is reached or until the environment is solved. The environment is considered solved when the average reward (over the last 100 episodes) is at least +13.



### Neural Network
The QNetwork model is 2 x 128 Fully Connected Layers with Relu activation followed by a final Fully Connected layer with the same number of units as the action size. The network has an initial dimension the same as the state size. 


### Hyper Parameters  

- n_episodes : chosed it to be 1000 and the training will stop just when score reached 13 or higher 
- max_t : maximum number of timesteps per episode = 10000
- eps_start : starting value of epsilon, for epsilon-greedy action selection = 0.5
- eps_end : minimum value of epsilon = 0.1
- eps_decay : multiplicative factor (per episode) for decreasing epsilon = 0.98
- BUFFER_SIZE : replay buffer size = int(1e6)
- BATCH_SIZ : mini batch size = 128
- GAMMA : discount factor = 0.99
- TAU : for soft update of target parameters = 1e-3 
- LR : learning rate for optimizer = 0.0001 
- UPDATE_EVERY : how often to update the network = 2



## Performance plot

![Reward Plot](https://github.com/helmogey/banana_collector/blob/master/plot.png?raw=true)


## Future Work
changing the fully connected layer to be convolutional layer then fully connected layer and the input will be the frame as a picture.  


