import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

# Select relevant columns
selected_data = data[
    ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]
]

# Define predictors (X) and target variable (y)
target = "class"
X = selected_data.drop(columns=[target])
y = selected_data[target]

# Split dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define a pipeline: Imputation -> Scaling -> Logistic Regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # Handle missing values
    ('scaler', StandardScaler()),                 # Standardize features
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'))
])

# Perform 10-fold cross-validation on the training set
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
cv_mean_accuracy = np.mean(cv_scores)
cv_std_accuracy = np.std(cv_scores)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on test data
y_test_pred = pipeline.predict(X_test)

# Confusion Matrix
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):")
print(conf_matrix_test)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=sorted(y.unique()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification Report for the Test Set
test_report = classification_report(y_test, y_test_pred)
print("\nClassification Report (Test Set):")
print(test_report)

# ROC Curve and AUC for Multiclass
y_test_binarized = label_binarize(y_test, classes=sorted(y.unique()))  # One-hot encode classes
y_test_probs = pipeline.predict_proba(X_test)  # Predicted probabilities for each class
n_classes = y_test_binarized.shape[1]

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i+1} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
