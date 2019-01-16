[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

### This repo contains solution for Tennis Unity Environment as my final project for udacity nanodegree(DeepRL) program.

### Problem:

A pair of agents learn to play tennis. Aim is to make both the tennis agent learn to hit the ball over the net. I have tried to solve this by using DDGP DeppRL aglorithm.

![Trained Agent][image1]

For more details please check out this link: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

### Obseration Space of length 24

### Actions:

This environment has an continuous action space size of 2.

### Rewards:

	- +0.1 To agent when hitting ball over net.
	- -0.1 To agent who let ball hit their ground, or hit ball out of bounds.

To solve this environment, we need to acheive a average score of 0.5.

### Environment Dependencies File:
	requirements.txt
	use command : pip install -r requirements.txt


### Unity Environment:
You need only select the environment that matches your operating system:

	- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
	- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
	- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
	- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

### Main File: Tennis.ipynb

### Running Trained Agent File: Tennis_Trained.py

### Saved Weights File: actor0.pth , actor1.pth , critic.pth
