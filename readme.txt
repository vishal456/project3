This repo contains solution for Tennis Unity Environment as my final project for udacity nanodegree(DeepRL) program.

Problem: 
A pair of agents learn to play tennis. Aim is to make both the tennis agent learn to hit the ball over the net. I have tried to solve this by using DDGP DeppRL aglorithm.

For more details please check out this link: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

Actions:
This environment has an continuous action space size of 2.

Rewards:
	+0.1 To agent when hitting ball over net.
	-0.1 To agent who let ball hit their ground, or hit ball out of bounds.

To solve this environment, we need to acheive a average score of 0.5.

Dependencies File:
	requirements.txt
	use command : pip install -r requirements.txt

Main File: Tennis.ipynb
Running Trained Agent File: Tennis_Trained.py
Saved Weights File: actor0.pth , actor1.pth , critic.pth
