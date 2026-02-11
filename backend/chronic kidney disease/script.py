from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path

# TODO: Importing libraries

warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("------ All libraries imported! ------\n")

# TODO: Loading the dataset

train = pd.read_csv('./chronic kidney disease/train.csv')
test = pd.read_csv('./chronic kidney disease/test.csv')

# --- Dataset-specific cleaning ---
# Remove two garbage rows at the top (metadata)
train = train.iloc[2:].reset_index(drop=True)

# Drop unnecessary columns if present
if 'S.no' in test.columns:
    test = test.drop('S.no', axis=1)
if 'class' in train.columns:
    train = train.drop('class', axis=1)
if 'class' in test.columns:
    test = test.drop('class', axis=1)

print("----- DATA OVERVIEW -----")
print(f"Training samples: {train.shape[0]}")
print(f"Features: {train.shape[1]}")
print(f"Test samples: {test.shape[0]} \n")

# TODO: Find target column

# Print Column names
print("Column names:")
for i, col in enumerate(train.columns):
    print(f"{i+1}. {col}")

# Define target column
target_col = 'affected'

# Ensure target is numeric
train[target_col] = pd.to_numeric(train[target_col], errors='coerce')
test[target_col] = pd.to_numeric(test[target_col], errors='coerce')

print(f"\nTARGET: {target_col} \n")
print(train[target_col].value_counts())
print("\n\n")

# TODO: Check dataset quality

# Check for missing values
print("----- MISSING VALUES -----")
missing = train.isnull().sum()
if missing.sum() > 0:
    print("\nMissing values found:")
    print(missing[missing > 0])
else:
    print("\n No missing values!")

# Statistics for numeric columns
print("\n------ STATISTICAL SUMMARY (numeric) -----")
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
if numeric_cols:
    print(train[numeric_cols].describe())

print("\n------ CATEGORICAL COLUMNS -----")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print("Categorical columns found:", categorical_cols)
else:
    print("No categorical columns.")

print()

# TODO: Fill missing values and encode categorical features

train_clean = train.copy()
test_clean = test.copy()

# Store encoders for later use if needed
encoders = {}

# Process all feature columns (both numeric and categorical)
feature_cols = [col for col in train.columns if col != target_col]

for col in feature_cols:
    # ---- Missing value imputation ----
    if pd.api.types.is_numeric_dtype(train_clean[col]):
        # Numeric: fill with median
        median_val = train_clean[col].median()
        train_clean[col].fillna(median_val, inplace=True)
        test_clean[col].fillna(median_val, inplace=True)
    else:
        # Categorical: fill with mode
        if train_clean[col].mode().empty:
            mode_val = "Unknown"
        else:
            mode_val = train_clean[col].mode()[0]
        train_clean[col].fillna(mode_val, inplace=True)
        test_clean[col].fillna(mode_val, inplace=True)

    # ---- Encode categorical variables ----
    if not pd.api.types.is_numeric_dtype(train_clean[col]):
        le = LabelEncoder()
        # Convert to string to avoid dtype issues
        train_vals = train_clean[col].astype(str)
        test_vals = test_clean[col].astype(str)
        # Fit on all unique values from both sets
        all_vals = pd.concat([train_vals, test_vals]).unique()
        le.fit(all_vals)
        train_clean[col] = le.transform(train_vals)
        test_clean[col] = le.transform(test_vals)
        encoders[col] = le
        # print(f"Encoded {col}")  # optional

print("----- Data cleaned and encoded! -----\n")

# TODO: Separate features and target

X = train_clean[feature_cols]
y = train_clean[target_col]

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"----- DATA SPLIT -----")
print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")

# TODO: Scale the features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n----- Features scaled! -----\n")

# TODO: Train models and evaluate

print("----- TRAINING MODELS -----\n")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=100, early_stopping=True)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    try:
        model.fit(X_train_scaled, y_train)
        y_pred_val = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, y_pred_val)
        results[name] = {'model': model, 'accuracy': val_acc}
        print(f"  ✓ Validation Accuracy: {val_acc:.4f}\n")
    except Exception as e:
        print(f"  ✗ Failed to train {name}: {e}\n")

print("----- All models trained! -----\n")

# TODO: Select best model and predict on test set

if not results:
    raise ValueError("No models trained successfully.")

# Create comparison table
print("----- MODEL COMPARISON -----")
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results]
}).sort_values('Accuracy', ascending=False)
print(comparison)

# Get best model
best_model_name = comparison.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"Accuracy: {comparison.iloc[0]['Accuracy']:.4f}")

# Evaluate best model
y_pred = best_model.predict(X_val_scaled)

print(f"\n📊 DETAILED EVALUATION: {best_model_name}\n")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# TODO: Show feature importance (if available)

print("\n🔍 FEATURE IMPORTANCE:\n")
if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance)
elif hasattr(best_model, 'coef_'):
    # For linear models
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': best_model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    print(importance)
else:
    print("No feature importance available for this model.")

# TODO: Prepare test data

X_test = test_clean[feature_cols]
X_test_scaled = scaler.transform(X_test)

# Make predictions
test_predictions = best_model.predict(X_test_scaled)

print(f"\n🎯 TEST PREDICTIONS")
print(f"Predictions made: {len(test_predictions)}")
print(f"Distribution:\n{pd.Series(test_predictions).value_counts()}")

# TODO: Save best model

print("\n💾 SAVING MODEL...\n")

# Create models directory
models_dir = Path(__file__).resolve().parent / 'models'
models_dir.mkdir(exist_ok=True)

# Save best model as chronic_kidney_disease.pkl
model_path = models_dir / 'chronic_kidney_disease.pkl'
joblib.dump(best_model, model_path)
print(f"✓ Saved best model ({best_model_name}) to {model_path}")

print("\n----- Model saved successfully! -----")