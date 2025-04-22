import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn import tree

# Load dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path, dtype={'column_name': str}, low_memory=False)


# Select relevant features
selected_data = data[
    ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]
]

target = "class"
X = selected_data.drop(columns=[target])
y = selected_data[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

# Train model
pipeline.fit(X_train, y_train)

# Extract trained Random Forest model
rf_model = pipeline.named_steps['classifier']

# Create directory to save decision tree images
tree_dir = "decision_trees"
os.makedirs(tree_dir, exist_ok=True)

# Plot individual decision trees
for i, tree_in_forest in enumerate(rf_model.estimators_[:5]):  # Limit to first 5 trees
    dotfile = os.path.join(tree_dir, f"dtree_{i}.dot")
    pngfile = os.path.join(tree_dir, f"dtree_{i}.png")
    
    tree.export_graphviz(tree_in_forest, 
                         out_file=dotfile, 
                         feature_names=X_train.columns, 
                         class_names=[str(cls) for cls in sorted(y_train.unique())], 
                         filled=True, 
                         rounded=True, 
                         max_depth=3)
    
    os.system(f"dot -Tpng {dotfile} -o {pngfile}")

print(f"Decision trees saved in '{tree_dir}' directory.")
