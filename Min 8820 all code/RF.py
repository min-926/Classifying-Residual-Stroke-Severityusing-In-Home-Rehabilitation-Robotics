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

#########################################################################
######################CHANGE ME##########################################
#########################################################################
# Load dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

#########################################################################
######################CHANGE ME##########################################
#########################################################################
selected_data = data[
    ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]
]

#########################################################################
######################CHANGE ME##########################################
#########################################################################
target =  "class"
X = selected_data.drop(columns=[target])
y = selected_data[target]

# Split dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train)
print(X_test)
# Define a pipeline: Imputation -> Scaling -> Random Forest
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # Handle missing values
    ('scaler', StandardScaler()),                 # Standardize features
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Perform 10-fold cross-validation on the training set
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
cv_mean_accuracy = np.mean(cv_scores)
cv_std_accuracy = np.std(cv_scores)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on training and test data
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Confusion matrix and metrics for the training set
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
train_report = classification_report(y_train, y_train_pred, output_dict=True)

conf_matrix_test = confusion_matrix(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred, output_dict=True)

# Extract required metrics
precision_weighted = train_report['weighted avg']['precision'] * 100
recall_weighted = train_report['weighted avg']['recall'] * 100
f1_weighted = train_report['weighted avg']['f1-score'] * 100

precision_weighted_test = test_report['weighted avg']['precision'] * 100
recall_weighted_test = test_report['weighted avg']['recall'] * 100
f1_weighted_test = test_report['weighted avg']['f1-score'] * 100

# Extract all unique class labels dynamically
unique_classes = sorted(y_train.unique())
class_metrics = {}

for cls in unique_classes:
    cls_str = str(cls)
    if cls_str in train_report:
        class_metrics[cls_str] = {
            'Precision (%)': test_report[cls_str]['precision'] * 100,
            'Recall (%)': test_report[cls_str]['recall'] * 100,
            'F1-Score (%)': test_report[cls_str]['f1-score'] * 100,
        }
    else:
        class_metrics[cls_str] = {
            'Precision (%)': 0.0,
            'Recall (%)': 0.0,
            'F1-Score (%)': 0.0,
        }

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# Compute AUC-ROC curve for multi-class classification
y_test_binarized = label_binarize(y_test, classes=unique_classes)
y_test_proba = pipeline.predict_proba(X_test)
n_classes = len(unique_classes)

# Plot AUC-ROC Curve
plt.figure(figsize=(6, 5))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {unique_classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve (Multi-Class)')
plt.legend(loc='lower right')
plt.show()

# Print results
print(f"Cross-Validation Accuracy: {cv_mean_accuracy * 100:.2f}% ± {cv_std_accuracy * 100:.2f}%")
print("\nConfusion Matrix (Training Set):")
print(conf_matrix_train)
print("\nPerformance Metrics (Weighted Average):")
print(f"Precision: {precision_weighted:.2f}%, Recall: {recall_weighted:.2f}%, F1-Score: {f1_weighted:.2f}%")
print("\nPerformance Metrics for Each Class:")
print(f"{'Class':<10} {'Precision (%)':<15} {'Recall (%)':<15} {'F1-Score (%)':<15}")
print("-" * 55)
for cls, metrics in class_metrics.items():
    print(f"{cls:<10} {metrics['Precision (%)']:<15.2f} {metrics['Recall (%)']:<15.2f} {metrics['F1-Score (%)']:<15.2f}")
