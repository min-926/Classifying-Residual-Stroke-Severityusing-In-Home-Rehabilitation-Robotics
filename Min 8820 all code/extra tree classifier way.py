


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Read the dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

# Define required columns
required_columns = ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]

# Check if all required columns exist
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"The dataset is missing the following required columns: {missing_columns}")

# Select relevant columns
selected_data = data[required_columns]

# Handle missing values
print("Handling missing values...")
selected_data = selected_data.fillna(selected_data.median(numeric_only=True))  # Fill numeric values with median
selected_data = selected_data.fillna(selected_data.mode().iloc[0])  # Fill categorical values with mode

# Define predictors and target
target = "class"
predictors = [col for col in selected_data.columns if col != target]

# Convert target variable to integer encoding if categorical
if selected_data[target].dtype.name in ["category", "object"]:
    selected_data[target] = selected_data[target].astype("category").cat.codes

X = selected_data[predictors]
y = selected_data[target]

# Scale predictors for better performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and holdout sets (80-20 split)
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the Extra Trees Classifier
extra_trees = ExtraTreesClassifier(
    n_estimators=150,  # Increased number of trees
    random_state=42,
    max_depth=None,  # No depth limitation
    max_features="sqrt",  # Use sqrt(features) at each split
    n_jobs=-1,  # Use all available processors
)

# Perform 10-fold cross-validation on the training set
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(extra_trees, X_train, y_train, cv=cv, scoring="accuracy")

# Train the model on the full training set
extra_trees.fit(X_train, y_train)

# Evaluate on the holdout set
y_holdout_pred = extra_trees.predict(X_holdout)

# Calculate performance metrics
accuracy = accuracy_score(y_holdout, y_holdout_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_holdout, y_holdout_pred, average="weighted"
)

# Confusion matrix
conf_matrix = confusion_matrix(y_holdout, y_holdout_pred)

# Output results
print("\nCross-Validation Results:")
print(f"Mean Accuracy on Training Set (%): {cv_scores.mean() * 100:.2f}")
print(f"Accuracy Standard Deviation: {cv_scores.std() * 100:.2f}")

print("\nHoldout Set Performance Metrics:")
print(f"Accuracy (%): {accuracy * 100:.2f}")
print(f"Precision (%): {precision * 100:.2f}")
print(f"Recall (%): {recall * 100:.2f}")
print(f"F1-Score (%): {f1_score * 100:.2f}")

print("\nConfusion Matrix:")
print(conf_matrix)

# Detailed classification report
print("\nDetailed Classification Report on Holdout Set:")
print(classification_report(y_holdout, y_holdout_pred))