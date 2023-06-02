from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np
import random

def generateHist(randomNums, n_bins, m, a, c):    
    fig, ax = plt.subplots(figsize=(10, 7))

    counts, bins, bars = ax.hist(randomNums, 
                                    bins=n_bins, 
                                    color='blue', 
                                    alpha=0.5)
    plt.legend([f'Bins: {n_bins}'])
    if m == 0:
        ax.set_title(f'Python Mersene Twister')
    else:
        ax.set_title(f'M:{m}, a:{a}, c:{c}')
    plt.show()
    
    return counts

def scatterTest(randomNums, a, c, m):
        
    # Scatter plot with size variations
    fig, ax = plt.subplots(figsize=(10, 7))
    x = randomNums[:-1]  # Take all numbers except the last one
    y = randomNums[1:]   # Shift the numbers by one position


    ax.scatter(x, y,  s= 4, color='green', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if m == 0:
        ax.set_title(f'Python Mersene Twister')
    else:
        ax.set_title(f'M:{m}, a:{a}, c:{c}')
    plt.show()

def chiSquareTest(observed_frequencies, samples, classes, alpha):
    print("\nBeginning Chi squared Test")

    expected_frequencies = [samples / classes] * classes

    # Calculate the critical value for the chi-square test
    df = classes - 1
    critical_value = stats.chi2.ppf(1 - alpha, df).round(2) 
    print(f"Df:{df} \nSign. lvl:{alpha} \nCrit. value:{critical_value}")
    

    # Calculate the test statistic
    t = 0
    for observed, expected in zip(observed_frequencies, expected_frequencies):
        t += ((observed - expected) ** 2) / expected
    print("Test statistic:", t)
    
    if abs(t) < abs(critical_value):
        print("Success!")
    else:
        print("Rejected")

    return t

def kolmogorov_smirnov(random_numbers, m, alpha):
    print("\nBeginning kolmogorov_smirnov Test")

    # Empirical Distribution
    n = len(random_numbers)
    if m == 0:
        ecdf = np.sort(random_numbers)
    else:
        ecdf = np.sort(random_numbers / m)

    # Actual Distribution
    cdf = np.arange(1, n + 1) / n

    # Test statistic
    t = np.max(np.abs(ecdf - cdf))
    # Critical value 
    critical_value = stats.ksone.ppf(1-alpha, n)
    print(f"Crit.value:{critical_value} Test stat:{t}")

    if abs(t) < abs(critical_value):
        print("Success!")
    else:
        print("Rejected")

def above_below_test(random_numbers, sign_lvl):
        print("\nBeginning Above Below test")

        median = np.median(np.array(random_numbers))
        runs = 0
        above = random_numbers[0] > median

        for sample in random_numbers:
            if (sample > median and not above) or (sample <= median and above):
                runs += 1
                above = not above

        print(runs)
        na = len(random_numbers) / 2
        mean = (2 * na * na) / (na+na) + 1
        variance = 2*na*na*(2*na*na - na - na) / ((na+na)**2*(na+na-1))
        t = (runs - mean) / variance**0.5
        critical_value = stats.norm.ppf(1 - sign_lvl/2)
        print(f"Median:{median} runs:{runs}")
        print(f"Cri.Value:{critical_value} Test statistic:{t}")

        if abs(t) < abs(critical_value):
            print("Success")
        else:
            print("Failure")

        return runs

def up_down_runs_test(data, alpha=0.05):
    # Compute the number of runs
    n = len(data)
    data_diff = np.diff(data)
    mask = np.hstack([data_diff, 0]) * np.hstack([0, data_diff])
    runs = np.sum(mask <= 0) - 1

    # Compute the expected number of runs and its standard deviation
    num_pos = np.sum(data_diff > 0)
    num_neg = np.sum(data_diff < 0)
    expected_runs = 2.0 * num_pos * num_neg / n + 1
    runs_std_dev = np.sqrt((expected_runs - 1) * (expected_runs - 2) / (n - 1))

    # Compute the Z statistic
    z = (runs - expected_runs) / runs_std_dev

    # Determine the critical value for the given alpha
    z_critical = stats.norm.ppf(1 - alpha / 2)

    print(f"Crit:{z_critical} T:{z} Runs:{runs}")

    # Test hypothesis
    if abs(z) > z_critical:
        print(f'Failure')
    else:
        print(f'Success')

def corr(samples, alpha=0.05):
    x = np.array(samples[:-1])
    x_mean = np.mean(x)
    y = np.array(samples[1:])
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    
    correlation = numerator / denominator

    # Compute degrees of freedom
    df = len(x) - 2

    # Compute the t-statistic
    t_stat = correlation * np.sqrt(df / ((1.0 - correlation) * (1.0 + correlation)))

    # Compute the critical value
    critical_value = stats.t.ppf(1 - alpha/2, df)
    print(f'T:{t_stat} c:{critical_value} cor:{correlation}')

    if np.abs(t_stat) > critical_value:
        print(f"Failure")
    else:
        print(f"Success")

class LCG:
    def __init__(self, m, a, c):
        self.m = m
        self.c = c
        self.a = a
    
    def run(self, xo,  rns):
        # Create list to store the generated numbers
        randomNums = [0]* (rns)
        # Seed
        randomNums[0] = xo

        for i in range(1, rns):
            # Follow the linear congruential method
            randomNums[i] = ( (randomNums[i-1] * self.a) + self.c ) % self.m
        
        return randomNums


# Start of Experiment

# Define the LCGs
# lcg_s = LCG(m=100003 , a=317, c=491)
# lcg_m = LCG(m=1000003 , a=517, c=971)
# lcg_l = LCG(m=1000000007, a=48271, c=0)
# lcg_list = [lcg_s, lcg_m, lcg_l]
# seeds = [1234, 12345, 123456]

# Hyperparameters
classes = 10
sign_lvl = 0.05
sample = 10000

# for l,s in zip(lcg_list, seeds):
#     print("\n-Beginning Experiment for LCG")
#     print(f"a:{l.a} c:{l.c} m:{l.m} Seed:{s}")

#     # Run the LCG
#     gen_numbers = l.run(xo=s, rns=sample)

#     corr(gen_numbers)

#     # Plots
#     scatterTest(gen_numbers, m=l.m, a=l.a, c=l.c)
#     counts = generateHist(gen_numbers, classes, m=l.m, a=l.a, c=l.c)

#     # Chi squared test
#     chiSquareTest(counts, samples=sample, classes=classes, alpha=sign_lvl)

#     # kolmogorov_smirnov test
#     kolmogorov_smirnov(random_numbers=np.array(gen_numbers), m=l.m, alpha=sign_lvl)

#     # Above below test
#     above_below_test(random_numbers=gen_numbers, sign_lvl=sign_lvl)

#     # Up Down test
#     up_down_runs_test(np.array(gen_numbers), alpha=sign_lvl)

print("\n-Beginning Experiment for System RNG Mersene Twister")
random.seed(0.5)
system_numbers = [0.5] + ([random.random() for _ in range(10000)])[:-1]
corr(system_numbers)

# # Plots
# scatterTest(system_numbers, m=0, a=0, c=0)
# counts = generateHist(system_numbers, classes, m=0, a=0, c=0)

# # Chi squared test
# chiSquareTest(counts, samples=sample, classes=classes, alpha=sign_lvl)

# # kolmogorov_smirnov test
# kolmogorov_smirnov(random_numbers=np.array(system_numbers), m=0, alpha=sign_lvl)

# # Above below test
# above_below_test(random_numbers=system_numbers, sign_lvl=sign_lvl)

# # Up Down test
# up_down_runs_test(np.array(system_numbers), alpha=sign_lvl)


