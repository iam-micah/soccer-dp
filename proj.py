import pandas as pd
from pydp.algorithms.laplacian import BoundedSum, BoundedMean
import matplotlib.pyplot as plt
import numpy as np

# Load the injuries data
injuries_csv_path = 'data/injuries.csv'
injuries_df = pd.read_csv(injuries_csv_path)

# Convert 'severity' column to numeric
severity_mapping = {'minor': 1, 'moderate': 2, 'severe': 3}
injuries_df['severity'] = injuries_df['severity'].map(severity_mapping)

# Differential Privacy Parameters
epsilon = 1.0
delta = 1e-5  # Small delta for approximate differential privacy
lower_bound = 1  # Adjust as per the data characteristics
upper_bound = 3  # Adjust as per the data characteristics

# Function to apply DP mean


def dp_mean(data, epsilon, lower_bound, upper_bound):
    dp_mean = BoundedMean(epsilon, delta, lower_bound, upper_bound)
    for value in data:
        dp_mean.add_entry(value)
    return dp_mean.result()

# Function to apply DP sum


def dp_sum(data, epsilon, lower_bound, upper_bound):
    dp_sum = BoundedSum(epsilon, delta, lower_bound, upper_bound)
    for value in data:
        dp_sum.add_entry(value)
    return dp_sum.result()

# Visualization function


def plot_data(data, labels, title, file_name):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, data, color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.savefig(file_name)
    plt.close()


# Analyze data with and without DP
categories = ['injury_type', 'severity', 'player_id']
for category in categories:
    # Non-private analysis
    counts = injuries_df.groupby(category).size()
    plot_data(counts.values, counts.index,
              f'Total Number of Injuries by {category} (No DP)', f'injuries_by_{category}_no_dp.png')

    # Differentially private analysis using BoundedSum
    dp_counts = [dp_sum(injuries_df[injuries_df[category] == x].severity.values,
                        epsilon, lower_bound, upper_bound) for x in counts.index]
    plot_data(dp_counts, counts.index,
              f'Total Number of Injuries by {category} (DP)', f'injuries_by_{category}_dp.png')

    # Differentially private mean severity analysis
    mean_severity = injuries_df.groupby(category)['severity'].mean()
    dp_mean_severity = [dp_mean(injuries_df[injuries_df[category] == x].severity.values,
                                epsilon, lower_bound, upper_bound) for x in mean_severity.index]
    plot_data(dp_mean_severity, mean_severity.index,
              f'Mean Severity of Injuries by {category} (DP)', f'severity_mean_by_{category}_dp.png')
