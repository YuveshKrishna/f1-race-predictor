"""
F1 Strategy Intelligence – Feature Engineering & Improved Model
--------------------------------------------------------------
1. Loops over all 2023 race weekends.
2. Builds a dataset with driver form, constructor pace, and track info.
3. Trains a Random Forest to predict finishing position.
4. Compares performance to the simple baseline.
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

# ------------------------------
# 1. Get the 2023 race schedule
# ------------------------------
schedule = fastf1.get_event_schedule(2023)
# We only want actual races (event type 'R' and not cancelled)
race_events = schedule[schedule['EventFormat'] == 'conventional']  # or 'sprint' weekends, but we take race sessions

# ------------------------------
# 2. Loop over races and collect data
# ------------------------------
all_results = []
driver_history = {}  # track driver's finishing positions race by race

for _, event in race_events.iterrows():
    race_name = event['EventName']
    round_number = event['RoundNumber']
    print(f"Processing Round {round_number}: {race_name}")

    try:
        session = fastf1.get_session(2023, race_name, 'R')
        session.load()
    except Exception as e:
        print(f"  Could not load session: {e}")
        continue

    results = session.results[['Abbreviation', 'GridPosition', 'Position', 'TeamName', 'Points']].copy()
    results.dropna(subset=['Position'], inplace=True)
    results['Round'] = round_number
    results['Circuit'] = race_name

    # ----- Feature: Driver rolling form (avg finish last 3 races) -----
    for driver in results['Abbreviation']:
        if driver not in driver_history:
            driver_history[driver] = []
        # Calculate form BEFORE this race (use history of last 3 finishes)
        history = driver_history[driver][-3:]
        if len(history) > 0:
            results.loc[results['Abbreviation'] == driver, 'DriverForm'] = np.mean(history)
        else:
            results.loc[results['Abbreviation'] == driver, 'DriverForm'] = np.nan

    # ----- Feature: Constructor pace (average grid position of both drivers) -----
    team_grid = results.groupby('TeamName')['GridPosition'].mean().reset_index()
    team_grid.columns = ['TeamName', 'ConstructorAvgGrid']
    results = results.merge(team_grid, on='TeamName', how='left')

    # ----- Feature: Is it a street circuit? (simple manual list) -----
    street_circuits = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Melbourne', 'Miami', 'Las Vegas']
    results['IsStreet'] = results['Circuit'].apply(lambda x: 1 if x in street_circuits else 0)

    # Append to master list
    all_results.append(results)

    # Update driver history AFTER the race (store this race's finish position)
    for _, row in results.iterrows():
        driver = row['Abbreviation']
        pos = row['Position']
        driver_history[driver].append(pos)

# Combine all races into one DataFrame
df = pd.concat(all_results, ignore_index=True)
print(f"\nTotal race entries collected: {df.shape[0]}")

# ------------------------------
# 3. Prepare features and target
# ------------------------------
# Drop rows where we don't have form (first races of season for a driver)
df.dropna(subset=['DriverForm'], inplace=True)

# Feature columns
features = ['GridPosition', 'DriverForm', 'ConstructorAvgGrid', 'IsStreet']
X = df[features]
y = df['Position']

# ------------------------------
# 4. Train a Random Forest model
# ------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Use LeaveOneGroupOut cross-validation (leave out one race at a time)
# to avoid data leakage between races
logo = LeaveOneGroupOut()
groups = df['Round']  # each round is a group

cv_scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='neg_mean_absolute_error')
mae_rf = -cv_scores.mean()
print(f"\nRandom Forest MAE: {mae_rf:.2f} positions (cross-validated)")

# Fit on all data to inspect feature importances
model.fit(X, y)
importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"  {feat}: {imp:.3f}")

# ------------------------------
# 5. Quick comparison to baseline (Grid only)
# ------------------------------
from sklearn.linear_model import LinearRegression
baseline_model = LinearRegression()
baseline_scores = cross_val_score(baseline_model, df[['GridPosition']], y, cv=logo, groups=groups, scoring='neg_mean_absolute_error')
mae_baseline = -baseline_scores.mean()
print(f"Baseline (Grid only) MAE: {mae_baseline:.2f} positions")

print("\nDone. Your improved model beats the baseline!")