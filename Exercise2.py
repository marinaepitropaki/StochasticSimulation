#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:56:45 2023

@author: mep
"""

# Exercise 2

import random
import numpy as np
import math
import matplotlib.pyplot as plt

'''
def geometric_distribution(p, k_values):

    q = 1 - p  # Probability of failure in a single trial
    # list, probability mass function
    pmf = [p * (q ** (k - 1)) for k in k_values]
    
    return pmf

def geometric_distribution(p, k_values):
    # Convert k_values to integers
    k_values = [int(k) for k in k_values]

    # Generate the geometric distribution
    pmf = [p * ((1 - p) ** (k - 1)) for k in k_values]

    return pmf
'''
def geometric_distribution(p, k_values):
    pmf =[]
    for k in range(len(k_values)):
        x = math.floor(math.log(k_values[k], 1-p))+1
        pmf.append(x)
    return pmf


def compare_geometric_histograms(generated, expected, n_bins):
    
    plt.hist(generated, bins=n_bins, density=True, alpha=0.5, color='b', label='Generated')
    plt.hist(expected, bins=n_bins, density=True, alpha=0.5, color='r', label='Expected')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Comparison of Distributions')
    plt.legend()
    plt.show()

def compare_six_point_histograms(crude, rejection, alias, n_bins):
    
    plt.hist([crude, rejection, alias], bins=n_bins, density=True, alpha=0.5, color=['b', 'r', 'g'], label=["Crude","Rejection", "Alias"])
    #plt.hist(expected, bins=n_bins, density=True, alpha=0.5, color='r', label='Expected')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Comparison of Distributions')
    plt.legend()
    plt.show()

def simulate_six_point_distribution_crude(p_i, q_i, X):
    outcomes = []
    for _ in range(len(X)):
        r = random.random()
        cumulative_sum = 0
        for i in range(len(p_i)):
            cumulative_sum += p_i[i]
            if cumulative_sum >= r:
                outcomes.append(i+1)
                break
    return outcomes

def simulate_six_point_distribution_rejection(p_i, q_i, X):
    outcomes = []
    M = max([p / q for p, q in zip(p_i, q_i)])  # Calculate the upper bound M
    while len(outcomes) < len(X):
        r = random.random()
        k = random.randint(1, 6)
        if r <= p_i[k-1] / (M * q_i[k-1]):
            outcomes.append(k)
    return outcomes
    

def simulate_six_point_distribution_alias(p_i, X):
    num_outcomes = len(p_i)

    target_prob = [p * num_outcomes for p in p_i]

    alias = [0] * num_outcomes
    prob = [0] * num_outcomes

    small_prob = []
    large_prob = []

    for i, p in enumerate(target_prob):
        if p < 1:
            small_prob.append(i)
        else:
            large_prob.append(i)

    while small_prob and large_prob:
        j = small_prob.pop()
        q = large_prob.pop()

        prob[j] = target_prob[j]
        alias[j] = q

        remaining_prob = target_prob[q] - (1 - target_prob[j])
        target_prob[q] = remaining_prob

        if remaining_prob < 1:
            small_prob.append(q)
        else:
            large_prob.append(q)

    outcomes = []
    for _ in range(len(X)):
        j = random.randint(0, num_outcomes - 1)
        r = random.random()

        if r < prob[j]:
            outcomes.append(j + 1)
        else:
            outcomes.append(alias[j] + 1)

    return outcomes

p= 0.5
size = 10000

#Generate 10000 pseudo-random numbers
X = np.random.random(size)

# Convert random numbers to the appropriate range for the geometric distribution
#k_values = np.ceil(np.log(X) / np.log(1 - p))
    
z_exp = np.random.geometric(p=p, size=size)
z_gen = geometric_distribution(p=p, k_values=X)
compare_geometric_histograms(z_gen, z_exp, max(max(z_exp), max(z_gen)) - 1)

p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
X_i = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

outcomes_crude = simulate_six_point_distribution_crude(p_i, X_i, X)

outcomes_rejection = simulate_six_point_distribution_rejection(p_i, X_i, X)

outcomes_alias = simulate_six_point_distribution_alias(p_i, X)

compare_six_point_histograms(outcomes_crude, outcomes_rejection, outcomes_alias, max(max(z_exp), max(z_gen)) - 1)

