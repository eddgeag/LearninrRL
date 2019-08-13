#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:34:20 2018

@author: edmond
"""

import gym
import time

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.reset()
    env.render()
    env.step(env.action_space.sample()) # take a random action
    time.sleep(0.02)
env.close()
#import gym
#env = gym.make('CartPole-v0')
#print(env.action_space)
##> Discrete(2)
#print(env.observation_space)