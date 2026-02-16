from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings("ignore")

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

print("------ All libraries imported! ------\n")

# TODO: Loading the dataset

train = pd.read_csv("./diabetes/train.csv")
test = pd.read_csv("./diabetes/test.csv")

print("----- DATA OVERVIEW -----")
print(f"Training samples: {train.shape[0]}")
print(f"Features: {train.shape[1]}")
print(f"Test samples: {test.shape[0]} \n")

print("----- FEATURE SUMMARY -----")
feature_summary = pd.DataFrame(
    {
        "Type": train.dtypes.astype(str),
        "Unique": train.nunique(),
        "Missing": train.isnull().sum(),  # Added missing values check
        "Missing %": (train.isnull().sum() / len(train) * 100).round(
            2
        ),  # Added percentage for clarity
    }
)

# Sort by Type to group similar features (optional)
print(feature_summary.sort_values(by="Type"))
print("\n")

# TODO: Find target column

# Print Column names
print("Column names:")
for i, col in enumerate(train.columns):
    print(f"{i+1}. {col}")

# Define target column
target_col = "diagnosed_diabetes"

# # Show target distribution
print(f"\nTARGET: {target_col} \n")
print(train[target_col].value_counts())
print("\n\n")
