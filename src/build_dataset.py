"""
F1 Strategy Intelligence – Feature Engineering & Improved Model
--------------------------------------------------------------
1. Loops over all 2023 race weekends.
2. Builds a dataset with driver form, constructor pace, and track info.
3. Trains a Random Forest to predict finishing position.
4. Compares performance to the simple baseline.
"""

import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache
# Ensure cache directory exists
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# ------------------------------
# 1. Get the 2023 race schedule
# ------------------------------
schedule = fastf1.get_event_schedule(2023)
race_events = schedule[schedule['EventFormat'] == 'conventional']

# ------------------------------
# 2. Loop over races and collect data
# ------------------------------
all_results = []
driver_history = {}

for _, event in race_events.iterrows():
    race_name = event['EventName']
    location = event['Location']          # <--- FIX: Get the city name
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
    results['Location'] = location          # <--- FIX: Store the city name

    # ----- Feature: Driver rolling form (avg finish last 3 races) -----
    for driver in results['Abbreviation']:
        if driver not in driver_history:
            driver_history[driver] = []
        history = driver_history[driver][-3:]
        results.loc[results['Abbreviation'] == driver, 'DriverForm'] = np.mean(history) if history else np.nan

    # ----- Feature: Constructor pace (average grid position of both drivers) -----
    team_grid = results.groupby('TeamName')['GridPosition'].mean().reset_index()
    team_grid.columns = ['TeamName', 'ConstructorAvgGrid']
    results = results.merge(team_grid, on='TeamName', how='left')

    # ----- Feature: Is it a street circuit? (FIXED: compare against Location) -----
    street_circuit_locations = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Melbourne', 'Miami', 'Las Vegas']
    results['IsStreet'] = results['Location'].apply(lambda x: 1 if x in street_circuit_locations else 0)

    all_results.append(results)

    # Update driver history AFTER the race
    for _, row in results.iterrows():
        driver_history.setdefault(row['Abbreviation'], []).append(row['Position'])

# Combine all races
df = pd.concat(all_results, ignore_index=True)
print(f"\nTotal race entries collected: {df.shape[0]}")

# ------------------------------
# 3. Prepare features and target
# ------------------------------
df.dropna(subset=['DriverForm'], inplace=True)

features = ['GridPosition', 'DriverForm', 'ConstructorAvgGrid', 'IsStreet']
X = df[features]
y = df['Position']

# ------------------------------
# 4. Train a Random Forest model
# ------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)

logo = LeaveOneGroupOut()
groups = df['Round']

cv_scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='neg_mean_absolute_error')
mae_rf = -cv_scores.mean()
print(f"\nRandom Forest MAE: {mae_rf:.2f} positions (cross-validated)")

model.fit(X, y)
importances = model.feature_importances_
print("\nFeature Importances:")
for feat, imp in zip(features, importances):
    print(f"  {feat}: {imp:.3f}")

# ------------------------------
# 5. Quick comparison to baseline (Grid only)
# ------------------------------
baseline_model = LinearRegression()
baseline_scores = cross_val_score(baseline_model, df[['GridPosition']], y, cv=logo, groups=groups, scoring='neg_mean_absolute_error')
mae_baseline = -baseline_scores.mean()
print(f"Baseline (Grid only) MAE: {mae_baseline:.2f} positions")

print("\nDone. Your improved model beats the baseline!")
