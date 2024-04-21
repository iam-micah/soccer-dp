import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to set x-ticks for better visualization


def set_x_ticks(ax):
    labels = ax.get_xticklabels()
    if len(labels) > 10:  # More than 10 labels, we reduce the number shown
        every_nth = len(labels) // 10
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)


# Load the data
injuries_csv_path = 'data/injuries.csv'
players_csv_path = 'data/players.csv'
appearances_csv_path = 'data/appearances.csv'

injuries_df = pd.read_csv(injuries_csv_path)
players_df = pd.read_csv(players_csv_path)
appearances_df = pd.read_csv(appearances_csv_path)

# Convert 'date_of_birth' and 'date' to datetime
players_df['age'] = pd.Timestamp.now().year - \
    pd.to_datetime(players_df['date_of_birth'], errors='coerce').dt.year
injuries_df['injury_date'] = pd.to_datetime(
    injuries_df['injury_date'], errors='coerce')
appearances_df['date'] = pd.to_datetime(
    appearances_df['date'], errors='coerce')

# Map 'injury_type' to numeric codes
unique_injury_types = injuries_df['injury_type'].unique()
injury_type_mapping = {type_: idx for idx,
                       type_ in enumerate(unique_injury_types)}
injuries_df['injury_type_code'] = injuries_df['injury_type'].map(
    injury_type_mapping)

# Differential Privacy Settings
epsilons = {
    'injury_type': 0.1,
    'injury_duration': 0.2,
    'game_participation': 0.3,
    'performance_metrics': 0.4
}
sensitivities = {
    'injury_type': 1,
    'injury_duration': 10,
    'game_participation': 5,
    'performance_metrics': 3
}

# Apply Local Differential Privacy - Randomized Response for injuries


def randomized_response(value, prob):
    if np.random.rand() < prob:
        return value
    else:
        return np.random.choice(list(set(injury_type_mapping.values()) - {value}))


injuries_df['dp_injury_type_code'] = injuries_df['injury_type_code'].apply(
    lambda x: randomized_response(x, np.exp(epsilons['injury_type']) / (1 + np.exp(epsilons['injury_type']))))

# Aggregate data for injuries and visualize
original_counts = injuries_df['injury_type'].value_counts()
dp_counts = injuries_df['dp_injury_type_code'].map(
    {v: k for k, v in injury_type_mapping.items()}).value_counts()

# Visualization for Injury Counts
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(original_counts.index, original_counts.values, color='blue')
ax.set_title('Original Injury Counts')
set_x_ticks(ax)
plt.tight_layout()
plt.savefig('original_injury_counts.png')
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(dp_counts.index, dp_counts.values, color='red')
ax.set_title('Differentially Private Injury Counts')
set_x_ticks(ax)
plt.tight_layout()
plt.savefig('dp_injury_counts.png')
plt.close()

# Merge injuries data with players data and perform age-based analysis
injury_player_df = pd.merge(injuries_df, players_df, on='player_id')
injury_player_df['days_injured'] = (pd.to_datetime(injury_player_df['actual_recovery_date'], errors='coerce') -
                                    pd.to_datetime(injury_player_df['injury_date'], errors='coerce')).dt.days

# Group by age group and calculate average days injured
bins = [18, 25, 30, 35, 40, 50]
labels = ['18-24', '25-29', '30-34', '35-39', '40+']
injury_player_df['age_group'] = pd.cut(
    injury_player_df['age'], bins=bins, labels=labels, right=False)
average_injury_duration = injury_player_df.groupby('age_group')[
    'days_injured'].mean()
scale = sensitivities['injury_duration'] / epsilons['injury_duration']
dp_average_injury_duration = average_injury_duration.apply(
    lambda x: max(0, x + np.random.laplace(0, scale)))

# Visualization for Average Injury Duration by Age Group
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(average_injury_duration.index.astype(str),
       average_injury_duration.values, color='green')
ax.set_title('Original Average Duration of Injuries by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Average Days Injured')
set_x_ticks(ax)
plt.tight_layout()
plt.savefig('original_average_injury_duration.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(dp_average_injury_duration.index.astype(str),
       dp_average_injury_duration.values, color='red')
ax.set_title('Differentially Private Average Duration of Injuries by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Average Days Injured (DP Applied)')
set_x_ticks(ax)
plt.tight_layout()
plt.savefig('dp_average_injury_duration.png')
plt.close()

# Merge the dataframes on player_id for the impact of injuries on game participation
player_injury_appearance_df = pd.merge(pd.merge(
    players_df, injuries_df, on='player_id'), appearances_df, on='player_id')

# Filter appearances to those in the same year as the injury
player_injury_appearance_df['injury_year'] = player_injury_appearance_df['injury_date'].dt.year
player_injury_appearance_df['game_year'] = player_injury_appearance_df['date'].dt.year
filtered_df = player_injury_appearance_df[player_injury_appearance_df['injury_year']
                                          == player_injury_appearance_df['game_year']]

# Count appearances before and after the injury
filtered_df['before_injury'] = filtered_df['date'] < filtered_df['injury_date']
appearances_before_after = filtered_df.groupby(
    ['player_id', 'before_injury']).size().unstack(fill_value=0)

# Add Laplace noise to the appearance counts for differential privacy
scale = sensitivities['game_participation'] / epsilons['game_participation']
appearances_before_after['before_injury_dp'] = appearances_before_after[True].apply(
    lambda x: max(0, int(x + np.random.laplace(0, scale))))
appearances_before_after['after_injury_dp'] = appearances_before_after[False].apply(
    lambda x: max(0, int(x + np.random.laplace(0, scale))))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(appearances_before_after.index.astype(str),
       appearances_before_after[True], color='blue', label='Before Injury')
ax.bar(appearances_before_after.index.astype(str),
       appearances_before_after[False], bottom=appearances_before_after[True], color='green', label='After Injury')
ax.set_title('Original Impact of Injuries on Game Participation')
ax.set_xlabel('Player ID')
ax.set_ylabel('Number of Appearances')
ax.legend()
set_x_ticks(ax)
plt.tight_layout()
plt.savefig('original_impact_of_injuries_on_game_participation.png')
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(appearances_before_after.index.astype(str),
       appearances_before_after['before_injury_dp'], color='blue', label='Before Injury (DP)')
ax.bar(appearances_before_after.index.astype(str),
       appearances_before_after['after_injury_dp'], bottom=appearances_before_after['before_injury_dp'], color='red', label='After Injury (DP)')
ax.set_title('DP Applied Impact on Game Participation')
ax.set_xlabel('Player ID')
ax.set_ylabel('Number of Appearances (DP Applied)')
ax.legend()
set_x_ticks(ax)
plt.tight_layout()
plt.savefig('dp_impact_of_injuries_on_game_participation.png')
plt.close()

# Calculate before and after injury performance for goals
goals_before_after = filtered_df.groupby(['player_id', 'before_injury'])[
    'goals'].sum().unstack(fill_value=0)

# Calculate before and after injury performance for assists
assists_before_after = filtered_df.groupby(['player_id', 'before_injury'])[
    'assists'].sum().unstack(fill_value=0)

# Applying DP for goals
scale = sensitivities['performance_metrics'] / epsilons['performance_metrics']
goals_before_after['before_injury_dp'] = goals_before_after[True].apply(
    lambda x: max(0, int(x + np.random.laplace(0, scale))))
goals_before_after['after_injury_dp'] = goals_before_after[False].apply(
    lambda x: max(0, int(x + np.random.laplace(0, scale))))

# Applying DP for assists
assists_before_after['before_injury_dp'] = assists_before_after[True].apply(
    lambda x: max(0, int(x + np.random.laplace(0, scale))))
assists_before_after['after_injury_dp'] = assists_before_after[False].apply(
    lambda x: max(0, int(x + np.random.laplace(0, scale))))

# Visualization for Goal Scoring Impact
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].bar(goals_before_after.index.astype(str),
           goals_before_after[True], color='blue', label='Goals Before Injury')
axs[0].bar(goals_before_after.index.astype(str), goals_before_after[False],
           bottom=goals_before_after[True], color='green', label='Goals After Injury')
axs[0].set_title('Original Impact of Injuries on Goals Scored')
axs[0].set_xlabel('Player ID')
axs[0].set_ylabel('Number of Goals')
axs[0].legend()
set_x_ticks(axs[0])
axs[1].bar(goals_before_after.index.astype(
    str), goals_before_after['before_injury_dp'], color='blue', label='Goals Before Injury (DP)')
axs[1].bar(goals_before_after.index.astype(str), goals_before_after['after_injury_dp'],
           bottom=goals_before_after['before_injury_dp'], color='red', label='Goals After Injury (DP)')
axs[1].set_title('DP Applied Impact on Goals Scored')
axs[1].set_xlabel('Player ID')
axs[1].set_ylabel('Number of Goals (DP Applied)')
axs[1].legend()
set_x_ticks(axs[1])
plt.tight_layout()
plt.savefig('impact_of_goals_on_game_participation_comparison.png')
plt.close()

# Visualization for Assist Impact
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].bar(assists_before_after.index.astype(
    str), assists_before_after[True], color='blue', label='Assists Before Injury')
axs[0].bar(assists_before_after.index.astype(str), assists_before_after[False],
           bottom=assists_before_after[True], color='green', label='Assists After Injury')
axs[0].set_title('Original Impact of Injuries on Assists')
axs[0].set_xlabel('Player ID')
axs[0].set_ylabel('Number of Assists')
axs[0].legend()
set_x_ticks(axs[0])
axs[1].bar(assists_before_after.index.astype(
    str), assists_before_after['before_injury_dp'], color='blue', label='Assists Before Injury (DP)')
axs[1].bar(assists_before_after.index.astype(str), assists_before_after['after_injury_dp'],
           bottom=assists_before_after['before_injury_dp'], color='red', label='Assists After Injury (DP)')
axs[1].set_title('DP Applied Impact on Assists')
axs[1].set_xlabel('Player ID')
axs[1].set_ylabel('Number of Assists (DP Applied)')
axs[1].legend()
set_x_ticks(axs[1])
plt.tight_layout()
plt.savefig('impact_of_assists_on_game_participation_comparison.png')
plt.close()
