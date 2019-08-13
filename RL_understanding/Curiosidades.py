#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:58:30 2019

@author: edmond
"""

import numpy as np
import matplotlib.pyplot as plt


sin=np.sin(np.arange(0,2+np.pi,0.0000001))
Amplitud=1
freq=0.0000001
T=1/freq
rango_temporal=np.arange(0,2+np.pi,0.0000001)
plt.plot(rango_temporal,sin)