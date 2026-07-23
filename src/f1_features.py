"""
f1_features.py
---------------
Shared feature-engineering pipeline for the F1 finishing-position model.

Both build_dataset.py (experimentation) and train_and_save.py (final model)
import build_season_dataset() from here, so fixes only need to be made once.
"""

import os
import fastf1
import pandas as pd
import numpy as np

# ---------- Cache Setup ----------
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# ---------- Constants ----------
# Street circuits – matched against FastF1's `Location` field (city name),
# NOT `EventName` (which is "X Grand Prix"). This is the fixed version.
STREET_CIRCUIT_LOCATIONS = {
    'Monaco', 'Singapore', 'Baku', 'Jeddah', 'Melbourne', 'Miami', 'Las Vegas'
}

# The exact feature order the model expects. Keep this consistent!
FEATURE_NAMES = ['GridPosition', 'DriverForm', 'ConstructorForm', 'IsStreet']


# ---------- Helper Functions ----------
def get_race_events(year, include_sprints=True):
    """
    Return the schedule rows for actual races in a season.

    include_sprints=False keeps only 'conventional' weekends, dropping
    sprint weekends. Sprint weekends have different qualifying/grid
    structures, so mixing them without a flag can confuse the model.
    """
    schedule = fastf1.get_event_schedule(year)
    if include_sprints:
        return schedule[schedule['EventFormat'] != 'testing']
    return schedule[schedule['EventFormat'] == 'conventional']


def build_season_dataset(year, include_sprints=True, verbose=True):
    """
    Loop over a season's races and build the modeling dataset.

    Returns:
        df: DataFrame with all features and the target (Position)
        feature_names: list of feature columns in the correct order
    """
    race_events = get_race_events(year, include_sprints=include_sprints)

    all_results = []
    driver_history = {}       # driver -> list of past finishing positions
    constructor_history = {}  # team -> list of past finishing positions (both cars)

    for _, event in race_events.iterrows():
        race_name = event['EventName']
        location = event['Location']
        round_number = event['RoundNumber']

        if verbose:
            print(f"Processing Round {round_number}: {race_name}")

        try:
            session = fastf1.get_session(year, race_name, 'R')
            session.load()
        except Exception as e:
            if verbose:
                print(f"  Could not load session: {e}")
            continue

        # Grab results
        results = session.results[['Abbreviation', 'GridPosition', 'Position', 'TeamName']].copy()
        results.dropna(subset=['Position'], inplace=True)
        results['Round'] = round_number
        results['Circuit'] = race_name
        results['Location'] = location

        # ----- Driver rolling form: avg finish over the last 3 races (BEFORE this one) -----
        driver_form = []
        for driver in results['Abbreviation']:
            history = driver_history.get(driver, [])[-3:]
            driver_form.append(np.mean(history) if history else np.nan)
        results['DriverForm'] = driver_form

        # ----- Constructor rolling form: avg finish of BOTH cars over the last 3 races -----
        # Uses the last 6 entries (2 cars x 3 races) for each team.
        constructor_form = []
        for team in results['TeamName']:
            history = constructor_history.get(team, [])[-6:]
            constructor_form.append(np.mean(history) if history else np.nan)
        results['ConstructorForm'] = constructor_form

        # ----- Street circuit indicator (FIXED: uses Location, not EventName) -----
        results['IsStreet'] = results['Location'].apply(
            lambda x: 1 if x in STREET_CIRCUIT_LOCATIONS else 0
        )

        all_results.append(results)

        # Update history AFTER the race to prevent data leakage
        for _, row in results.iterrows():
            driver_history.setdefault(row['Abbreviation'], []).append(row['Position'])
            constructor_history.setdefault(row['TeamName'], []).append(row['Position'])

    # Combine everything into one DataFrame
    df = pd.concat(all_results, ignore_index=True)

    if verbose:
        print(f"\nTotal race entries collected: {df.shape[0]}")

    # Drop rows where we don't have form data (early-season races)
    df.dropna(subset=['DriverForm', 'ConstructorForm'], inplace=True)

    return df, FEATURE_NAMES
