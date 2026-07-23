"""
train_and_save.py
-----------------
Trains the Random Forest model using the shared feature pipeline (f1_features.py).
Saves the model and feature names for the Streamlit dashboard.
"""
import os
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

# ---------- Fix Paths (so we can import f1_features from the parent folder) ----------
# Get the root directory (parent of the 'src' folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add root to Python path so we can import f1_features
sys.path.append(ROOT_DIR)

# ---------- Import the shared feature pipeline ----------
from f1_features import build_season_dataset, FEATURE_NAMES

# ---------- Build the dataset ----------
print("🔄 Building dataset from 2023 season...")
df, feature_names = build_season_dataset(2023, include_sprints=True)

print(f"\n📊 Dataset ready: {df.shape[0]} entries")
print(f"📋 Features: {feature_names}")

# ---------- Define features and target ----------
X = df[feature_names]
y = df['Position']
groups = df['Round']

# ---------- Train final model on full dataset ----------
print("\n🧠 Training Random Forest model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------- Evaluate (Leave-One-Race-Out CV) ----------
print("📈 Evaluating with Leave-One-Race-Out cross-validation...")
logo = LeaveOneGroupOut()
cv_scores = cross_val_score(
    model, X, y, 
    cv=logo, 
    groups=groups, 
    scoring='neg_mean_absolute_error'
)
mae = -cv_scores.mean()
print(f"✅ Cross-validated MAE: {mae:.2f} positions")

# ---------- Feature Importances ----------
importances = model.feature_importances_
print("\n📊 Feature Importances:")
for feat, imp in zip(feature_names, importances):
    print(f"  {feat}: {imp:.3f}")

# ---------- Save model and feature names ----------
# Save in the ROOT_DIR so the dashboard (which looks in ../) can find them
model_path = os.path.join(ROOT_DIR, 'f1_model.pkl')
features_path = os.path.join(ROOT_DIR, 'feature_names.pkl')

joblib.dump(model, model_path)
joblib.dump(feature_names, features_path)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Feature names saved to: {features_path}")
print("\n🚀 You can now run the Streamlit dashboard:")
print("   streamlit run dashboard/app.py")
