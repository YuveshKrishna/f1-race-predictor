"""
train_and_save.py
-----------------
1. Loads the whole 2023 F1 season via FastF1.
2. Creates features (GridPosition, DriverForm, ConstructorAvgGrid, IsStreet).
3. Trains a Random Forest model.
4. Saves model and feature names to disk.
"""

import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import joblib
import warnings
warnings.filterwarnings('ignore')

# Cache FastF1 data (this folder only stores raw API responses)
# Ensure cache directory exists
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# ------------------------------
# 1. Get 2023 schedule and loop races
# ------------------------------
schedule = fastf1.get_event_schedule(2023)
race_events = schedule[schedule['EventFormat'] == 'conventional']

all_results = []
driver_history = {}

for _, event in race_events.iterrows():
    race_name = event['EventName']
    round_number = event['RoundNumber']
    print(f"Processing {race_name} (Round {round_number})")

    try:
        session = fastf1.get_session(2023, race_name, 'R')
        session.load()
    except Exception as e:
        print(f"  Skipping: {e}")
        continue

    results = session.results[['Abbreviation', 'GridPosition', 'Position', 'TeamName']].copy()
    results.dropna(subset=['Position'], inplace=True)
    results['Round'] = round_number
    results['Circuit'] = race_name

    # Driver form: average finish position in last 3 races
    for driver in results['Abbreviation']:
        if driver not in driver_history:
            driver_history[driver] = []
        history = driver_history[driver][-3:]
        results.loc[results['Abbreviation'] == driver, 'DriverForm'] = np.mean(history) if history else np.nan

    # Constructor average grid position
    team_grid = results.groupby('TeamName')['GridPosition'].mean().reset_index()
    team_grid.columns = ['TeamName', 'ConstructorAvgGrid']
    results = results.merge(team_grid, on='TeamName', how='left')

    # Street circuit indicator
    street_circuits = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Melbourne', 'Miami', 'Las Vegas']
    results['IsStreet'] = results['Circuit'].apply(lambda x: 1 if x in street_circuits else 0)

    all_results.append(results)

    # Update history after race
    for _, row in results.iterrows():
        driver_history[row['Abbreviation']].append(row['Position'])

# Combine all races
df = pd.concat(all_results, ignore_index=True)
df.dropna(subset=['DriverForm'], inplace=True)

print(f"\nDataset ready: {df.shape[0]} entries")

# ------------------------------
# 2. Define features and target
# ------------------------------
feature_names = ['GridPosition', 'DriverForm', 'ConstructorAvgGrid', 'IsStreet']
X = df[feature_names]
y = df['Position']

# ------------------------------
# 3. Train final model on full dataset
# ------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Quick evaluation (leave-one-race-out CV)
logo = LeaveOneGroupOut()
cv_scores = cross_val_score(model, X, y, cv=logo, groups=df['Round'], scoring='neg_mean_absolute_error')
print(f"Cross‑validated MAE: {-cv_scores.mean():.2f} positions")

# ------------------------------
# 4. Save model and feature names
# ------------------------------
joblib.dump(model, 'f1_model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("\nModel saved as 'f1_model.pkl'")
print("Feature names saved as 'feature_names.pkl'")
print("You can now run the Streamlit dashboard!")
