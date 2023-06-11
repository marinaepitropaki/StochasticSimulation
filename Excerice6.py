import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2, rv_discrete

# Define the target distribution
def unorm_P(i, m, A):
    if 0 <= i <= m:
        return (A ** i) / np.math.factorial(i)
    else:
        return 0 # out of range, reject

def G(i, j, m, A):
    return unorm_P(i, m, A) * unorm_P(j, m, A) if i + j <= m else 0

def hastings_part1(iterations, A, m):
    # Initialize
    i_current = np.random.randint(0, m+1) 

    running_means = []
    samples = []

    for _ in range(iterations):
        # Propose a new state
        i_proposal = i_current + np.random.choice([-1, 1])
        
        # Calculate the acceptance probability
        accept_prob = min(1, unorm_P(i_proposal, m, A) / unorm_P(i_current, m, A))
        
        # Decide whether to accept the proposed state
        if np.random.rand() <= accept_prob:
            i_current = i_proposal 
        
        samples.append(i_current)

        # Running mean
        running_mean = np.mean(samples)
        running_means.append(running_mean)

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.hist(samples, bins=range(m+2), density=True, alpha=0.5)
    plt.title('Histogram of Samples')

    # Plot the running mean
    plt.subplot(1, 2, 2)
    plt.plot(running_means)
    plt.title('Running Mean of Samples')
    plt.show()

    return samples

def hastings_part2(n_iterations, m, A):

    # Initialize
    i, j = 0, 0
    samples = [(i, j)]
    for _ in range(n_iterations):
        # Propose a new state
        di, dj = np.random.choice([-1, 0, 1], 2)
        i_proposed, j_proposed = i + di, j + dj

        if 0 <= i_proposed <= m and 0 <= j_proposed <= m and i_proposed + j_proposed <= m:
            # Calculate acceptance probability
            accept_prob = min(1, G(i_proposed, j_proposed, m, A) / G(i, j, m, A))

            # Accept or reject the proposed move
            if np.random.uniform() < accept_prob:
                i, j = i_proposed, j_proposed

        samples.append((i, j))

    return samples

def hastings_part2_coord(n_iterations, m, A):
    # Initialize i, j
    i, j = 0, 0

    # Store samples
    samples = []

    for _ in range(n_iterations):
        # Random walk proposal for i and j
        iprop = i + np.random.choice([-1, 0, 1])
        jprop = j + np.random.choice([-1, 0, 1])
        
        # Ensure we stay within bounds
        iprop = max(0, min(m, iprop))
        jprop = max(0, min(m, jprop))

        # Compute acceptance probability
        if iprop + jprop <= m:
            accept_prob = min(1, G(iprop, j, m, A) / G(i, j, m, A))
            if np.random.uniform() < accept_prob:
                i = iprop
        
        if iprop + jprop <= m:
            accept_prob = min(1, G(i, jprop, m, A) / G(i, j, m, A))
            if np.random.uniform() < accept_prob:
                j = jprop
        
        # Store sample
        samples.append((i, j))

    return samples

def part1_run():
    m = 10
    A = 8
    iterations = 10000
    samples = hastings_part1(iterations=iterations, m=m, A=A)

    # Perform chi squared test
    c = 1 / sum([unorm_P(i, m, A) for i in range(m+1)]) # calculate c for P

    burn_in_samples = 100
    expected_frequencies = np.array([(len(samples[burn_in_samples:]))*c * unorm_P(i, m, A) for i in range(m+1)])
    observed_frequencies, _ = np.histogram(samples[burn_in_samples:], bins=range(m+2), density=False)
    chi_square_statistic = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)
    df = m-1  # degrees of freedom
    critical_value = chi2.ppf(0.95, df)  # 95% confidence level
    print(f"Chi-square statistic: {chi_square_statistic}")
    print(f"Critical value: {critical_value}")
    if chi_square_statistic > critical_value:
        print("We reject the null hypothesis")
    else:
        print("We do not reject the null hypothesis")

    # Actual probabilities
    blocked_lines = [i for i in range(m+1)]
    p_values = [c * unorm_P(i, m, A) for i in range(m+1)]

    plt.bar(blocked_lines, p_values, width=0.7)
    plt.xlabel('Blocked Lines')
    plt.ylabel('Probability')
    plt.title('P probability dsitribution')
    plt.show()

def part2b_run():
    A=4
    m = 10 
    n_iterations = 100000
    samples = hastings_part2_coord(n_iterations, m, A)

    # 2d histogram
    plt.figure(figsize=(8,8))
    plt.hist2d(samples[:,0], samples[:,1], bins=(range(11), range(11)), cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel('i')
    plt.ylabel('j')
    plt.title('2D histogram of i and j')
    plt.show()

    # Running mean
    running_means_i = np.cumsum(samples[:,0]) / (np.arange(len(samples[:,0])) + 1)
    running_means_j = np.cumsum(samples[:,1]) / (np.arange(len(samples[:,1])) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(running_means_i, color='r', label='i')
    plt.plot(running_means_j, color='b', label='j')
    plt.title("Running mean of i and j")
    plt.xlabel("Iteration")
    plt.ylabel("Running mean")
    plt.legend(loc='lower right')
    plt.show()

    # Chi squared test
    burn_in_samples = 5000
    samples = np.array(samples[burn_in_samples:]) 

    # Calcualte c
    c_sum = 0
    for i in range(m+1):
        for j in range(m+1-i):
            c_sum += ((8 ** i) / np.math.factorial(i)) * ((8 ** j) / np.math.factorial(j))
    c= 1 / c_sum

    # Calculate the expected frequencies
    expected_frequencies = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1-i):
            expected_frequencies[i, j] = c * ((8 ** i) / np.math.factorial(i)) * ((8 ** j) / np.math.factorial(j))

    # Calculate the observed frequencies
    observed_frequencies = np.zeros((m+1, m+1))
    for i, j in samples:
        observed_frequencies[i, j] += 1
    for i in range(m+1):
        for j in range(m+1):
            observed_frequencies[i, j] = observed_frequencies[i, j] / (n_iterations-burn_in_samples)

    expected_frequencies = expected_frequencies.flatten()
    observed_frequencies = observed_frequencies.flatten()

    valid_bins = expected_frequencies > 0
    expected_frequencies = expected_frequencies[valid_bins]
    observed_frequencies = observed_frequencies[valid_bins]

    chi_square_statistic = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)
    df = 11*11 - 1   # degrees of freedom
    critical_value = chi2.ppf(0.95, df)  # 95% confidence level
    print(f"Chi-square statistic: {chi_square_statistic}")
    print(f"Critical value: {critical_value}")
    if chi_square_statistic > critical_value:
        print("We reject the null hypothesis")
    else:
        print("We do not reject the null hypothesis")

def part2a_run():
    A=4
    m = 10 
    n_iterations = 100000

    samples = hastings_part2(n_iterations, m, A)

    # 2d histogram
    plt.figure(figsize=(8,8))
    plt.hist2d(samples[:,0], samples[:,1], bins=(range(11), range(11)), cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel('i')
    plt.ylabel('j')
    plt.title('2D histogram of i and j')
    plt.show()

    # Running mean
    running_means_i = np.cumsum(samples[:,0]) / (np.arange(len(samples[:,0])) + 1)
    running_means_j = np.cumsum(samples[:,1]) / (np.arange(len(samples[:,1])) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(running_means_i, color='r', label='i')
    plt.plot(running_means_j, color='b', label='j')
    plt.title("Running mean of i and j")
    plt.xlabel("Iteration")
    plt.ylabel("Running mean")
    plt.legend(loc='lower right')
    plt.show()

    # Chi squared test
    burn_in_samples = 5000
    samples = np.array(samples[burn_in_samples:]) 

    # Calcualte c
    c_sum = 0
    for i in range(m+1):
        for j in range(m+1-i):
            c_sum += ((8 ** i) / np.math.factorial(i)) * ((8 ** j) / np.math.factorial(j))
    c= 1 / c_sum

    # Calculate the expected frequencies
    expected_frequencies = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1-i):
            expected_frequencies[i, j] = c * ((8 ** i) / np.math.factorial(i)) * ((8 ** j) / np.math.factorial(j))

    # Calculate the observed frequencies
    observed_frequencies = np.zeros((m+1, m+1))
    for i, j in samples:
        observed_frequencies[i, j] += 1
    for i in range(m+1):
        for j in range(m+1):
            observed_frequencies[i, j] = observed_frequencies[i, j] / (n_iterations-burn_in_samples)

    expected_frequencies = expected_frequencies.flatten()
    observed_frequencies = observed_frequencies.flatten()

    valid_bins = expected_frequencies > 0
    expected_frequencies = expected_frequencies[valid_bins]
    observed_frequencies = observed_frequencies[valid_bins]

    chi_square_statistic = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)
    df = 11*11 - 1   # degrees of freedom
    critical_value = chi2.ppf(0.95, df)  # 95% confidence level
    print(f"Chi-square statistic: {chi_square_statistic}")
    print(f"Critical value: {critical_value}")
    if chi_square_statistic > critical_value:
        print("We reject the null hypothesis")
    else:
        print("We do not reject the null hypothesis")






# Part 1
#part1_run()

# Part 2.a
# part2a_run()

# Part 2.b
#part2b_run()

# Part 2.c
#part2b_run()
