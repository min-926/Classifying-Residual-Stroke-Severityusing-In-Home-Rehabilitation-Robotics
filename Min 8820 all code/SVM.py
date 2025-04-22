import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)

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

# Handle missing values explicitly before passing into pipeline
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)  # Ensure all NaNs are replaced before model training

# Verify missing values are handled
print(f"Total missing values after imputation: {np.isnan(X).sum()}")  # Should print 0

# Binarize target labels for multi-class ROC
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# Split dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define a pipeline: Scaling -> OneVsRest LinearSVC Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', OneVsRestClassifier(LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=42)))
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

# Confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Test Set")
plt.show()

# Get decision function scores for ROC curve
y_test_bin = label_binarize(y_test, classes=np.unique(y))
decision_scores = pipeline.named_steps['classifier'].decision_function(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], decision_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), decision_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC Curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in enumerate(colors[:n_classes]):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], linestyle="--", color="black", label=f"Micro-Avg (AUC = {roc_auc['micro']:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Print results
print(f"Cross-Validation Accuracy: {cv_mean_accuracy * 100:.2f}% ± {cv_std_accuracy * 100:.2f}%")
print("\nConfusion Matrix (Test Set):")
print(conf_matrix_test)
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))
