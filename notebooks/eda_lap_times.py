"""
F1 Strategy Intelligence – First Steps
--------------------------------------
Loads the 2023 Bahrain Grand Prix, explores lap times for Verstappen,
and trains a simple Linear Regression model to predict finishing position
from grid position. Run this in your terminal or import into a notebook.
"""

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ------------------------------
# 1. Enable FastF1 caching
# ------------------------------
fastf1.Cache.enable_cache('cache')

# ------------------------------
# 2. Load the 2023 Bahrain GP race session
# ------------------------------
print("Loading 2023 Bahrain Grand Prix race data...")
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()
print(f"Session loaded: {session.event}")
print(f"Laps: {session.laps.shape[0]}, Results: {session.results.shape[0]}")

# ------------------------------
# 3. Explore lap times for a specific driver
# ------------------------------
laps = session.laps
driver_laps = laps.pick_driver('VER').pick_quicklaps()

plt.figure(figsize=(12, 5))
sns.scatterplot(
    data=driver_laps,
    x='LapNumber',
    y='LapTime',
    hue='Compound',
    palette='viridis'
)
plt.title('Verstappen Lap Times – Bahrain 2023')
plt.ylabel('Lap Time (seconds)')
plt.xlabel('Lap Number')
plt.tight_layout()
plt.show()

# ------------------------------
# 4. Build a baseline prediction model
#    (GridPosition → Finishing Position)
# ------------------------------
results = session.results[['Abbreviation', 'GridPosition', 'Position']].copy()
results.dropna(inplace=True)  # Remove drivers who did not start/finish

X = results[['GridPosition']]   # Feature
y = results['Position']         # Target

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nTest MAE (Grid→Finish): {mae:.2f} positions")

# Cross-validation (more robust)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross‑validated MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} positions")

print("\nAll done. You've just built your first F1 prediction model!")