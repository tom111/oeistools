#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sympy.ntheory import isprime
from scipy import stats

"""
This script plots the occurrence counts of numbers in the OEIS (Online Encyclopedia of Integer Sequences) data.
It reads a trimmed CSV file containing these counts and performs statistical analysis and visualization.

Following the orginal coloring in Giuglimetti's graph:
- Categorize numbers as perfect powers, highly composite numbers, primes, and others numbers.
    - perfect powers: green
    - highly composite numbers: yellow
    - primes: red
    - regular numbers: blue
- Perform a linear regression on the log-transformed counts of positive numbers.
- Plot the occurrence counts on a logarithmic scale, with different markers for each category of numbers.
- Add a regression line to the plot to show the relationship between number size and occurrence count.
- Save the plot as a PNG file.
"""

def is_perfect_power(n):
    """Check if n is a perfect power (square, cube, etc.)"""
    if n < 2:
        return False
    # Check powers up to log2(n)
    max_exp = int(np.log2(n)) + 1
    for exp in range(2, max_exp + 1):
        root = round(n ** (1/exp))
        if root ** exp == n:
            return True
    return False

def get_highly_composite_numbers(limit):
    """Get all highly composite numbers up to limit"""
    # Pre-computed set of highly composite numbers up to 100K
    # Source: OEIS A002182
    hcn = {1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440, 83160}
    return {x for x in hcn if x <= limit}

# Read the trimmed CSV file
df = pd.read_csv('occurrence_counts_trimmed.csv')

# Create arrays for different number types
numbers = df['Number'].values
counts = df['Count'].values

# Calculate regression (using only positive numbers)
mask = numbers > 0
x = np.log(numbers[mask])
y = np.log(counts[mask])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create masks for different number types
perfect_power_mask = [is_perfect_power(n) for n in numbers]
hcn_set = get_highly_composite_numbers(max(numbers))
highly_composite_mask = [n in hcn_set for n in numbers]
prime_mask = [isprime(n) for n in numbers]
regular_mask = ~(np.array(perfect_power_mask) | np.array(highly_composite_mask) | np.array(prime_mask))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot data
bigmarkersize = 3
smallmarkersize = 1
plt.semilogy(numbers[perfect_power_mask], counts[perfect_power_mask], 'g.', markersize=bigmarkersize, label='Perfect powers')
plt.semilogy(numbers[highly_composite_mask], counts[highly_composite_mask], 'y.', markersize=bigmarkersize, label='Highly composite')
plt.semilogy(numbers[prime_mask], counts[prime_mask], 'r.', markersize=bigmarkersize, label='Primes')
plt.semilogy(numbers[regular_mask], counts[regular_mask], 'b.', markersize=smallmarkersize, label='Regular numbers')

# Add regression line
x_range = np.linspace(1, max(numbers), 1000)
y_fit = np.exp(intercept + slope * np.log(x_range))
plt.plot(x_range, y_fit, 'k-', label=f'Fit: log(count) = {slope:.2f}*log(n) + {intercept:.2f}')

# Add labels and title
plt.xlabel('Number')
plt.ylabel('Count (log scale)')
plt.title('OEIS Sequence Occurrence Counts')
plt.grid(True)
plt.legend()

# Print regression statistics
print(f"Regression results:")
print(f"log(Count) = {slope:.3f} * log(n) + {intercept:.3f}")
print(f"R-squared: {r_value**2:.3f}")

# Save the plot
plt.savefig('occurrence_counts_plot.png', dpi=300, bbox_inches='tight')
plt.close() 
print("Plot saved to occurrence_counts_plot.png")