import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time

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

def compare_distribution_histograms(crude, rejection, alias, n_bins):
    
    plt.figure(figsize=(10, 8))
    plt.hist([crude, rejection, alias], bins=n_bins, density=True, alpha=0.5, color=['b', 'r', 'g'], label=["Crude","Rejection", "Alias"])
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Comparison of Distributions')
    plt.legend()
    plt.show()

def simulate_crude(p_i, size):
    outcomes = []
    for _ in range(size):
        r = random.random()
        cumulative_sum = 0
        for i in range(len(p_i)):
            cumulative_sum += p_i[i]
            if cumulative_sum >= r:
                outcomes.append(i+1)
                break

    return outcomes

def simulate_rejection(p_i, q_i, size):
  outcomes = []
  M = max([p / q for p, q in zip(p_i, q_i)])  # Calculate the upper bound M
  while len(outcomes) < size:
      r = random.random()
      k = random.choices(range(1, 7), q_i)[0]  # Choose k with probability q_i
      if r <= p_i[k-1] / (M * q_i[k-1]):
          outcomes.append(k)
  return outcomes
    

def simulate_alias(p_i, size):
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
    for _ in range(size):
        j = random.randint(0, num_outcomes - 1)
        r = random.random()

        if r < prob[j]:
            outcomes.append(j + 1)
        else:
            outcomes.append(alias[j] + 1)

    return outcomes


p = 0.5
size = 10000
# Generate 10000 pseudo-random numbers
X = np.random.random(size)

# Geometric distribution
z_exp = np.random.geometric(p=p, size=size)
z_gen = geometric_distribution(p=p, k_values=X)
compare_geometric_histograms(z_gen, z_exp, max(max(z_exp), max(z_gen)) - 1)

# Different methods
p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
q_i = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

start = time.time()
outcomes_crude = simulate_crude(p_i, size) # crude
end = time.time()
print(end-start)

start = time.time()
outcomes_rejection = simulate_rejection(p_i, q_i, size) # rejection
end = time.time()
print(end-start)

start = time.time()
outcomes_alias = simulate_alias(p_i, size) # alias
end = time.time()
print(end-start)

compare_distribution_histograms(outcomes_crude, outcomes_rejection, outcomes_alias, 6)
