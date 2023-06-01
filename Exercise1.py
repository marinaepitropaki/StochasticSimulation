#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:50:59 2023

@author: mep
"""
from matplotlib import pyplot as plt
import numpy as np

#Exercise 1 - Stochastic Simulations
#%% LCG form scratch
#
class LCGFromScratch:
    # Seed value
    Xo = 5 #as low as we want, just the starting point
    
    # Modulus parameter
    m = 7 # sufficiently high like 50000 circle around the points
    
    # Multiplier term
    a = 3
    
    # Increment term
    c = 3
    
    # Number of Random numbers to be generated
    noOfRandomNums = 10000
    
    # To store random numbers
    randomNums = [0]* (noOfRandomNums)
    
    n_bins = 5

    def linearCongruentialMethod(self, Xo, m, a, c, randomNums, noOfRandomNums):
        
        # Initialize the seed state
        randomNums[0] = Xo
        
        # Traverse to generate required numbers of random numbers
        
        for i in range(1, noOfRandomNums):
            
            # Follow the linear congruential method
            randomNums[i] = ((randomNums[i-1]*a)+c) % m
        return randomNums
        
 #%%  Generate 10.000 (pseudo-) random numbers and present these numbers in a histogramme
    def generateHist(self, randomNums, n_bins):
        
        # Creating histogram
        fig, ax = plt.subplots(figsize=(10, 7))
        #n_bins = 5 #expected 10 bins
        counts, bins, bars = ax.hist(randomNums, 
                                     bins=n_bins, 
                                     color='blue', 
                                     alpha=0.5)
        plt.legend([f'Bins: {n_bins}'])
        plt.show()
        
        return counts, bins, bars

    
#%% Tests

    def scatterTest(self, randomNums):
        
        # Scatter plot with size variations
        fig, ax = plt.subplots(figsize=(10, 7))
        x = randomNums[:-1]  # Take all numbers except the last one
        y = randomNums[1:]   # Shift the numbers by one position
        
        
        ax.scatter(x, y,  s= 4, color='green', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Random Number Generator Scatter Plot')
        plt.show()
        
    def chiSquareTest(self, observed_frequencies, expected_frequencies):
        
        # Calculate the chi-square statistic
        chi_square = 0
        for observed, expected in zip(observed_frequencies, expected_frequencies):
            chi_square += ((observed - expected) ** 2) / expected

        # Calculate the critical value for the chi-square test
        # calculate it by hand
        critical_value = 16.92  # From chi-square distribution table with (n_bins - 1) degrees of freedom and significance level 0.05

        # Perform the chi-square test
        p_value = 1 if chi_square < critical_value else 0
        

        return chi_square, p_value
        

    def kolmogorov_smirnov_test(self, sample1, sample2, alpha):
        # graph expected
        
        # Sort the samples
        sample1 = np.sort(sample1)
        sample2 = np.sort(sample2)
        
        # Calculate the empirical cumulative distribution functions (ECDFs)
        n1 = len(sample1)
        n2 = len(sample2)
        ecdf1 = np.arange(1, n1 + 1) / n1
        ecdf2 = np.arange(1, n2 + 1) / n2
        
        # Calculate the test statistic (D)
        d = np.max(np.abs(ecdf1 - ecdf2))
        
        # Calculate the critical value
        critical_value = np.sqrt(-0.5 * np.log(alpha / 2) / (n1 + n2))
        
        # Perform the hypothesis test
        if d > critical_value:
            return "Reject null hypothesis. The samples are significantly different."
        else:
            return "Cannot reject null hypothesis. The samples are not significantly different."
    
    
    
    def above_below_test(self, data, median):
        # graph expected
        
        above_runs = 0
        below_runs = 0
        current_run = 1 if data[0] > median else -1
    
        for i in range(1, len(data)):
            if data[i] > median and current_run > 0:
                current_run += 1
            elif data[i] < median and current_run < 0:
                current_run -= 1
            else:
                if current_run > 0:
                    above_runs += 1
                else:
                    below_runs += 1
                current_run = 1 if data[i] > median else -1
    
        # Handle the final run
        if current_run > 0:
            above_runs += 1
        else:
            below_runs += 1
    
        n = above_runs + below_runs
        mean = (2 * above_runs * below_runs) / n + 1
        variance = (mean - 1) * (mean - 2) / (n - 1)
        z = (above_runs - mean) / variance**0.5
        p_value = 2 * (1 - self.calculate_cdf(abs(z)))
        return p_value

    def runTest(self):
        # Function Call
        self.linearCongruentialMethod(self.Xo, self.m, self.a, self.c, self.randomNums, self.noOfRandomNums)

        runs = 1
        n1, n2 = 0, 0
        for i in range(1, len(self.randomNums)):
            if self.randomNums[i] != self.randomNums[i - 1]:
                runs += 1
                if self.randomNums[i] > self.randomNums[i - 1]:
                    n1 += 1
                else:
                    n2 += 1
        n = n1 + n2
        mean = (2 * n1 * n2) / n + 1
        variance = (mean - 1) * (mean - 2) / (n - 1)
        z = (runs - mean) / variance**0.5
        p_value = 2 * (1 - self.calculate_cdf(abs(z)))

        return p_value
    
    def calculate_cdf(self, x):
        # Calculate the cumulative distribution function (CDF) of the standard normal distribution
        cdf = (1 + self.erf(x / np.sqrt(2))) / 2
        return cdf

    def erf(self, x):
        # Approximation of the error function (erf) using a series expansion
        erf = 0
        term = x
        sign = 1

        for k in range(1, 100):
            erf += sign * term / (2 * k - 1)
            sign *= -1
            term *= x ** 2 / k
        
        return erf
    
    def correlationTest(self, x, y):
        #graph expected
        
        # Calculate the correlation coefficient (Pearson's correlation)
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
        sum_x_squared = sum([xi ** 2 for xi in x])
        sum_y_squared = sum([yi ** 2 for yi in y])

        correlation = (n * sum_xy - sum_x * sum_y) / (np.sqrt(n * sum_x_squared - sum_x ** 2) * np.sqrt(n * sum_y_squared - sum_y ** 2))

        # Perform the t-test for correlation
        t = correlation * np.sqrt((n - 2) / (1 - correlation ** 2))
        p_value = 2 * (1 - self.calculate_cdf(abs(t)))

        return correlation, p_value


#%% Results 

#%% LCG Instance
        
# Create an instance of the LCGFromScratch class
lcg = LCGFromScratch()

#%% Histogram & Scatter

# Call the generateHist method

#bad choice
randomNums = lcg.linearCongruentialMethod(lcg.Xo, lcg.m, lcg.a, lcg.c, lcg.randomNums, lcg.noOfRandomNums)
counts, bins, bars = lcg.generateHist(randomNums, 5)
lcg.scatterTest(randomNums)
print("Counts: {},".format(counts))
expected_frequencies= [lcg.noOfRandomNums/5]*len(bins)
chi_square, p_value =lcg.chiSquareTest(counts, expected_frequencies)
print("Chi_square: {}, binary_p_value: {}".format(chi_square, p_value))
# Apply the Kolmogorov-Smirnov test
#alpha = 0.05
#result = lcg.kolmogorov_smirnov_test(counts, expected_frequencies, alpha)
#print("Kolmogorov: {},".format(result))

#bad choice
randomNums2 = lcg.linearCongruentialMethod(3, 50000, 5, 1, lcg.randomNums, lcg.noOfRandomNums)
counts2, bins2, bars2 = lcg.generateHist(randomNums2, 25)
lcg.scatterTest(randomNums2)
print("Counts2: {}".format(counts2))
expected_frequencies2= [lcg.noOfRandomNums/25]*len(bins)
chi_square2, p_value2 =lcg.chiSquareTest(counts2, expected_frequencies2)
print("Chi_square2: {}, binary_p_value2: {}".format(chi_square2, p_value2))

#good choice
randomNums3 = lcg.linearCongruentialMethod(3, 65536, 129, 26461, lcg.randomNums, 10000)
counts3, bins3, bars3 = lcg.generateHist(randomNums3, 25)
lcg.scatterTest(randomNums3)
print("Counts3: {}".format(counts3))
expected_frequencies3= [lcg.noOfRandomNums/25]*len(bins)
chi_square3, p_value3 =lcg.chiSquareTest(counts3, expected_frequencies3)
print("Chi_square3: {}, binary_p_value3: {}".format(chi_square3, p_value3))



#%% Kolmogorov

# Extract the generated random numbers
generated_nums = lcg.randomNums

# Split the generated_nums into two parts
observed_sample = generated_nums[:len(generated_nums)//2]
comparison_sample = generated_nums[len(generated_nums)//2:]

# Apply the Kolmogorov-Smirnov test
alpha = 0.05
result = lcg.kolmogorov_smirnov_test(observed_sample, comparison_sample, alpha)

print(result)

#%% run & above below

# Perform run test
p_value_run = lcg.runTest()
print("Run test p-value:", p_value_run)

# Generate random numbers for correlation test
x = lcg.randomNums[:100]  # Example subset of random numbers
y = lcg.randomNums[100:200]  # Example subset of random numbers

# Perform correlation test
correlation, p_value_corr = lcg.correlationTest(x, y)
print("Correlation coefficient:", correlation)
print("Correlation test p-value:", p_value_corr)

#%% C, repeat the process for a,b,M

best_combination = None
best_score = float('inf')

worst_combination = None
worst_score = 0

# Define the ranges for values of "a", "b", and "M" to experiment with
a_values = [1, 2, 3]
c_values = [1, 2, 3]
M_values = [100, 200, 300]










