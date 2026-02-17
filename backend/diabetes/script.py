from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

train = pd.read_csv('./diabetes/train.csv')
test = pd.read_csv('./diabetes/test.csv')

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
target_col = 'diagnosed_diabetes'

# # Show target distribution
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
    
# Statistics
print("\n------ STATISTICAL SUMMARY -----")

# Get numeric columns (exclude id and target)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
print(train[numeric_cols].describe())

# TODO: Fill missing values (if any)

train_clean = train.copy()
test_clean = test.copy()

for col in numeric_cols:
    if train_clean[col].isnull().sum() > 0:
        median_val = train_clean[col].median()
        train_clean[col].fillna(median_val, inplace=True)
        test_clean[col].fillna(median_val, inplace=True)
        print(f"Filled {col} with median: {median_val}")

print("----- Data cleaned! -----\n")

# TODO: Separate features and target

X = train_clean[numeric_cols]
y = train_clean[target_col]

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"----- DATA SPLIT -----")
print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")

# TODO: Train models and evaluate (with scaling in pipeline)

print("----- TRAINING MODELS -----\n")

# Define models wrapped in pipelines (scaler + model)
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
    ]),
    'LightGBM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ]),
}

results = {}

for name, pipeline in models.items():
    print(f"Training {name}...")

    # Train (pipeline handles scaling internally)
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_val = pipeline.predict(X_val)

    # Score
    val_acc = accuracy_score(y_val, y_pred_val)

    results[name] = {
        'model': pipeline,
        'accuracy': val_acc
    }

    print(f"  ✓ Validation Accuracy: {val_acc:.4f}\n")

print("----- All models trained! -----\n")

# TODO: Select best model and predict on test set

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

# Evaluate best model (pipeline handles scaling)
y_pred = best_model.predict(X_val)

print(f"\n📊 DETAILED EVALUATION: {best_model_name}\n")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# TODO: Show feature importance (if available)

print("\n🔍 FEATURE IMPORTANCE:\n")
# Get classifier from pipeline for feature importance
classifier = best_model.named_steps['classifier']
if hasattr(classifier, 'feature_importances_'):
    importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance': classifier.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance)

# TODO: Prepare test data

X_test = test_clean[numeric_cols]

# Make predictions (pipeline handles scaling)
test_predictions = best_model.predict(X_test)

print(f"\n🎯 TEST PREDICTIONS")
print(f"Predictions made: {len(test_predictions)}")
print(f"Distribution:\n{pd.Series(test_predictions).value_counts()}")

# TODO: Save best model

print("\n💾 SAVING MODEL...\n")

# Create models directory
models_dir = Path(__file__).resolve().parent / 'models'
models_dir.mkdir(exist_ok=True)

# Save best model as diabetes.pkl
model_path = models_dir / 'diabetes.pkl'
joblib.dump(best_model, model_path)
print(f"✓ Saved best model ({best_model_name}) to {model_path}")

print("\n----- Model saved successfully! -----")