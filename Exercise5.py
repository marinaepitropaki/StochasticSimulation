#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:43:05 2023

@author: mep
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t
#from Exercise3 import confidence_intervals

def crude_monte_carlo_integral(k_values):
    f_x = np.exp(k_values)
    mean = np.mean(f_x)
    var = np.var(f_x)
    confidence_interval = t.interval(alpha=0.95, df=len(f_x)-1,
              loc=np.mean(f_x),
              scale=stats.sem(f_x))
    
    return mean, var, confidence_interval

def antithetic_variables(k_values):
    a_values = 1-k_values
    f_x = (np.exp(k_values) + np.exp(a_values))/2
    mean = np.mean(f_x)
    var = np.var(f_x)
    confidence_interval = t.interval(alpha=0.95, df=len(f_x)-1,
              loc=np.mean(f_x),
              scale=stats.sem(f_x))
    
    return mean, var, confidence_interval

def control_variables(k_values):
    c = -0.14085 *12
    f_x = np.exp(k_values) + c * (k_values - 0.5)
    mean = np.mean(f_x)
    var = np.var(f_x)
    confidence_interval = t.interval(alpha=0.95, df=len(f_x)-1,
              loc=np.mean(f_x),
              scale=stats.sem(f_x))
    return mean, var, confidence_interval

size = 100
X = np.random.random(size)
mean, var, confidence_interval= crude_monte_carlo_integral(X)
a_mean, a_var, a_confidence_interval= antithetic_variables(X)
c_mean, c_var, c_confidence_interval= control_variables(X)

print(f'Crude Monte Carlo: \n confidence interval: {confidence_interval} \n mean: {mean}, \n variance: {var}')
print(f'Antithetic: \n confidence interval: {a_confidence_interval} \n mean: {a_mean}, \n variance: {a_var}')
print(f'Control Variables: \n confidence interval: {c_confidence_interval} \n mean: {c_mean}, \n variance: {c_var}')