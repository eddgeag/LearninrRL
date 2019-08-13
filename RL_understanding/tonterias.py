#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:34:02 2019

@author: edmond
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


freq = np.arange(0, 1,0.1)
theta = 2 * np.pi * freq

#sin=np.sin(theta)
#cos=np.cos(theta)
#plt.close('all')
#ax = plt.subplot(111, projection='polar')
#ax.plot(theta, sin**2+cos**2 ,'bo')
#freq = np.arange(0, 1,0.001)
#theta = 2 * np.pi * freq
#
#sin=np.sin(theta)
#cos=np.cos(theta)
#ax.plot(theta, sin**2+cos**2 ,'b-')
#
##ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
##ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
#ax.grid(False)
#
#ax.set_title("A line plot on a polar axis", va='bottom')
#plt.show()


#x=np.arange(0,10,1)
#y=np.ravel(np.ones(shape=(1,10)))
#l=plt.plot(x,y,'bo')
#plt.setp(l, markersize=5)
#l=plt.plot(x,y,'b-')
#
rewards_lista=[-1,2,6,3,2]

G=[]
g=0
for k, idx in enumerate(rewards_lista):
    T=len(rewards_lista)
    g=(-k+T)*(0.5**idx)
    G.append(g)
1/4*(0+0.9*2.3)+1/4*(0+0.9*0.7)+1/4*(0+0.9*0.4)+1/4*(0-0.9*0.4)