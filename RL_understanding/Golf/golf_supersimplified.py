#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:34:39 2018

@author: edmond
"""

import numpy as np


R=np.array([-1,-3,0]).reshape(3,1)
P=np.array([[0.5,0,0.5],[0.5,0.5,0],[0,0,1]])
M=np.linalg.inv((1-P))
v=M@R
