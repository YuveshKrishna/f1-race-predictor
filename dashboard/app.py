"""
f1_features.py
---------------
Shared feature-engineering pipeline for the F1 finishing-position model.

Both build_dataset.py (experimentation/ablations) and train_and_save.py
(final model used by the dashboard) import build_season_dataset() from
here, so a fix -- like the IsStreet bug below -- only has to be made in
one place instead of two.
"""

import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('cache')

# Circuits considered "street circuits" for the IsStreet feature.
#
# IMPORTANT: this is matched against FastF1's `Location` field (the
# city/circuit the event is held at), NOT `EventName`.
# `EventName` is the full "<Country> Grand Prix" name (e.g. "Azerbaijan
# Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix") and
# does not contain "Baku", "Jeddah", or "Melbourne" as substrings -- so
# checking EventName against this list silently produced IsStreet=0 for
# every single row. `Location` gives the city directly (Baku, Jeddah,
# Melbourne, Miami, Monaco, Singapore, Las Vegas), which is what this
# list is written against.
STREET_CIRCUIT_LOCATIONS = {
    'Monaco', 'Singapore', 'Baku', 'Jeddah', 'Melbourne', 'Miami', 'Las Vegas'
}

FEATURE_NAMES = ['GridPosition', 'DriverForm', 'ConstructorForm', 'IsStreet']


def get_race_events(year, include_sprints=True):
    """
    Return the schedule rows for actual races in a season.

    include_sprints=False keeps only 'conventional' weekends, dropping
    2023's ~6 sprint weekends. That's a deliberate scope decision, not
    an oversight: sprint weekends have a different qualifying/grid
    structure (sprint shootout grid vs. normal quali grid), and mixing
    them into training without a separate "is_sprint_weekend" flag
    would let the model conflate two different race formats under the
    same GridPosition feature.
    """
    schedule = fastf1.get_event_schedule(year)
    if include_sprints:
        return schedule[schedule['EventFormat'] != 'testing']
    return schedule[schedule['EventFormat'] == 'conventional']


def build_season_dataset(year, include_sprints=True, verbose=True):
    """
    Loop over a season's races and build the modeling dataset.

    Returns (df, feature_names) where feature_names is the fixed,
    ordered list of columns the model expects: FEATURE_NAMES above.
    """
    race_events = get_race_events(year, include_sprints=include_sprints)

    all_results = []
    driver_history = {}       # driver -> list of past finishing positions
    constructor_history = {}  # team   -> list of past finishing positions (both cars)

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

        results = session.results[['Abbreviation', 'GridPosition', 'Position', 'TeamName', 'Points']].copy()
        results.dropna(subset=['Position'], inplace=True)
        results['Round'] = round_number
        results['Circuit'] = race_name
        results['Location'] = location

        # ----- Driver rolling form: avg finish over the last 3 races, BEFORE this one -----
        driver_form = []
        for driver in results['Abbreviation']:
            history = driver_history.get(driver, [])[-3:]
            driver_form.append(np.mean(history) if history else np.nan)
        results['DriverForm'] = driver_form

        # ----- Constructor rolling form: avg finish (both cars) over recent races -----
        # This replaces the old ConstructorAvgGrid feature, which averaged
        # the two teammates' GRID positions in the SAME race. That made it
        # almost redundant with GridPosition itself (same-race, same
        # session) rather than telling the model anything new about recent
        # team pace. ConstructorForm instead looks at each team's last
        # 3 races' worth of FINISHING positions (last 6 entries = 2 cars
        # x 3 races), the same way DriverForm does for individual drivers.
        constructor_form = []
        for team in results['TeamName']:
            history = constructor_history.get(team, [])[-6:]
            constructor_form.append(np.mean(history) if history else np.nan)
        results['ConstructorForm'] = constructor_form

        # ----- Street circuit indicator (fixed: matched on Location, see above) -----
        results['IsStreet'] = results['Location'].apply(
            lambda x: 1 if x in STREET_CIRCUIT_LOCATIONS else 0
        )

        all_results.append(results)

        # Update history AFTER the race, so this race's own result never
        # leaks into its own features.
        for _, row in results.iterrows():
            driver_history.setdefault(row['Abbreviation'], []).append(row['Position'])
            constructor_history.setdefault(row['TeamName'], []).append(row['Position'])

    df = pd.concat(all_results, ignore_index=True)
    if verbose:
        print(f"\nTotal race entries collected: {df.shape[0]}")

    # Drop rows without form (first races of the season for a driver/team)
    df.dropna(subset=['DriverForm', 'ConstructorForm'], inplace=True)

    return df, FEATURE_NAMES
