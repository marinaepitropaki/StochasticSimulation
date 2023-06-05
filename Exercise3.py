#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:00:50 2023

@author: mep
"""

import numpy as np
from math import log, pi, cos, sin, sqrt
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

def exponential_distribution(p, k_values):
    
    exp_dist = []
    for k in range(len(k_values)):
        exp_dist.append(-log(k_values[k])/p)
    
    return exp_dist

def box_muller(k_values):
    z1 = []
    z2 =[]
    for k in k_values:
        R = np.random.random(1)
        z1.append(sqrt(-2*log(k))*cos(2*pi*R))
        z2.append(sqrt(-2*log(k))*sin(2*pi*R))

    return z1, z2

def central_limit(size, random_var):
    sample_means = []

    for _ in range(size):
        X = np.sum(np.random.uniform(0, 1, random_var)) - random_var / 2
        sample_means.append(X)

    return sample_means

def pareto_distribution(beta, k_values, k_experiment):
    z =[]
    for k in range(len(k_values)):
        z.append(beta*(k_values[k]**(-1/k_experiment)))
    
    return z

def distplot(data, kde=True, color=None, title = ''):

    sns.histplot(data, kde=kde, color = color)
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{title} Distribution Plot')
    plt.show()

p= 0.5
size = 10000
beta = 1
k_experiment = [2.05, 2.5, 3, 4]

#Generate 10000 pseudo-random numbers
X = np.random.random(size)
    
z_exp = exponential_distribution(p=p, k_values=X)
z_box1, z_box2 = box_muller(X)
z_clt = central_limit(size, random_var=10)




distplot(z_exp, title ='Exponential')
distplot(z_box1, title = 'Box Muller 1')
distplot(z_box2, color = 'r', title = 'Box Muller 2')
distplot(z_clt, color='g', title = 'Central Limit')

for k in range(len(k_experiment)):
    
    z_prt_k = pareto_distribution(beta, X, k_experiment[k])
    distplot(z_prt_k, title= f'Pareto k ={ k_experiment[k]}')

