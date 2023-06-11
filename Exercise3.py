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

def pareto_distribution(beta, r_nums, k):
    z =[]
    for n in r_nums:
        z.append(beta*(n**(-1/k)))
    
    return z

def pareto_stats(beta, k):
    mean = beta*(k/(k-1))
    var = (beta**2)*(k/(((k-1)**2)*(k-2)))
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

# Initialize
p = 0.5
size = 10000
beta = 1
k_experiment = [2.05, 2.5, 3, 4]
confidence = 0.95

#Generate 10000 uniform distributed numbers
X = np.random.random(size)

# Part 1.a
z_exp = exponential_distribution(p=p, k_values=X)
z_box1, z_box2 = box_muller(X)

distplot(z_exp, title ='Exponential')
distplot(z_box1, title = 'Box Muller 1')
distplot(z_box2, color = 'r', title = 'Box Muller 2')

z_prt =[]
for k in k_experiment:
    z_prt_k = pareto_distribution(beta, X, k)
    z_prt.append(z_prt_k)
    distplot(z_prt_k, title= f'Pareto k ={ k}')

# part 3
z_conf = confidence_intervals(z_box1, 10, 100, 0.95)
print(z_conf)


