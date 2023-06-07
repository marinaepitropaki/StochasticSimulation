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
from scipy.stats import t

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

def pareto_stats(beta, k_experiment):
    mean = beta*(k_experiment/(k_experiment-1))
    var = (beta**2)*(k_experiment/(((k_experiment-1)**2)*(k_experiment-2)))
    return mean, var

def confidence_intervals(normal_dist_data, n_obs, num_int, confidence):
    confidence_intervals = []
    
    
    for _ in range(num_int):
        sample = np.random.choice(normal_dist_data, size=n_obs, replace= False)
        mean = np.mean(sample)
        variance = np.var(sample, ddof=1)
        dof = n_obs-1
        t_crit = np.abs(t.ppf((1-confidence)/2,dof))
        # Calculate confidence interval
        interval_upper = mean+variance*t_crit/np.sqrt(variance / n_obs)
        interval_lower = mean-variance*t_crit/np.sqrt(variance / n_obs)
        confidence_intervals.append((interval_lower, interval_upper))
    
    return confidence_intervals
    
    
    
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
confidence = 0.95

#Generate 10000 pseudo-random numbers
X = np.random.random(size)
    
z_exp = exponential_distribution(p=p, k_values=X)
z_box1, z_box2 = box_muller(X)
z_clt = central_limit(size, random_var=10)

distplot(z_exp, title ='Exponential')
distplot(z_box1, title = 'Box Muller 1')
distplot(z_box2, color = 'r', title = 'Box Muller 2')
distplot(z_clt, color='g', title = 'Central Limit')

z_prt =[]
for k in range(len(k_experiment)):
    
    z_prt_k = pareto_distribution(beta, X, k_experiment[k])
    z_prt.append(z_prt_k)
    distplot(z_prt_k, title= f'Pareto k ={ k_experiment[k]}')
    
pareto_mean, pareto_var = pareto_stats(beta, k_experiment=2.05)
print(f'Pareto mean: {pareto_mean} \n Pareto variance: {pareto_var}')
np_mean = np.mean(z_prt[0])
np_var = np.var(z_prt[0])
print(f'Analytical mean: {np_mean} \n Analytical variance: {np_var}')

z_conf = confidence_intervals(z_box1, 10, 100, 0.95)
print(z_conf)

f_y = np.random.exponential(k_experiment[3], X)
#distplot(f_y, title = 'Pareto k = {k_experiment[3]} from composition')

pareto_composition = []
for i in f_y:
    x = exponential_distribution(i, np.random.random(1))
    pareto_composition.append(x[0])

distplot(pareto_composition)





