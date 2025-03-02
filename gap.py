#!/usr/bin/env python3

import os
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm  # Progress bar
import argparse

"""
This script processes the OEIS (Online Encyclopedia of Integer Sequences) data to analyze the occurrences of each number.
It traverses through sequence files, extracts numbers, and counts their occurrences.
The script outputs statistics including:
- The largest and smallest numbers found (including negative numbers).
- The number of times the integer '1' appears.
- The smallest non-negative integer not present in the OEIS data.

The results are saved to a CSV file, and if a cutoff is specified, a trimmed version of the data is also saved.
The script is designed to analyze the numbers that appear in the three lines printed on the OEIS web pages.
"""

DEFAULT_CUTOFF = 10000
csv_path = "occurrence_counts.csv"
trimmed_csv_path = "occurrence_counts_trimmed.csv"

# Path to the OEIS data folder
DATA_PATH = "./seq"

# Regular expression to extract integers (positive and negative)
number_pattern = re.compile(r"-?\d+")

# Dictionary to count occurrences
number_counts = Counter()

# Argument parsing, looking for cutoff value.
parser = argparse.ArgumentParser(description='Process OEIS sequence and find gaps')
parser.add_argument('-c', '--cutoff', type=int, help='Optional cutoff value for trimmed output', default=None)
args = parser.parse_args()

# Get total number of files for progress tracking
all_folders = [os.path.join(DATA_PATH, folder) for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]
all_files = [os.path.join(folder, filename) for folder in all_folders for filename in os.listdir(folder) if filename.endswith(".seq")]
total_files = len(all_files)

# Traverse OEIS directory with progress bar
with tqdm(total=total_files, desc="Processing Sequence Files", unit="files") as pbar:
    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Only process lines starting with %S, %T, or %U
                if line.startswith(("%S", "%T", "%U")):
                    numbers = number_pattern.findall(line)
                    number_counts.update(map(int, numbers))
        pbar.update(1)  # Update progress bar after each file

# Convert to DataFrame
df = pd.DataFrame(number_counts.items(), columns=["Number", "Count"])
df.sort_values(by="Number", inplace=True)

# Save full counts to CSV
df.to_csv(csv_path, index=False)

# Save trimmed counts
cutoff = args.cutoff if args.cutoff is not None else DEFAULT_CUTOFF
trimmed_df = df[
    (df["Number"] >= 0) & 
    (df["Number"] < cutoff)
].copy()
trimmed_df.to_csv(trimmed_csv_path, index=False)

# Get sorted array of numbers once
numbers = df["Number"].values  # numbers are already sorted from earlier
numbers = numbers[numbers >= 0]  # filter to non-negative numbers only

# Find first gap in sequence
missing_number = 0
for num in numbers:
    if num > missing_number:
        break
    missing_number = num + 1

# Print some stats:
largest_number = df['Number'].max()
smallest_number = df['Number'].min()
number_of_ones = number_counts[1]
print(f"Largest number occurring: {largest_number}")
print(f"Smallest number occurring (including negatives): {smallest_number}")
print(f"Number of ones occurring: {number_of_ones}")
print(f"Smallest non-negative integer not in the OEIS: {missing_number}")
print(f"Full count statistics saved to {csv_path}.")
print(f"Trimmed count statistics saved to {trimmed_csv_path}.")