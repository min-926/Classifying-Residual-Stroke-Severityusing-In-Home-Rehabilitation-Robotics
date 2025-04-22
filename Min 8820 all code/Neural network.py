import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Read the dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

# Select relevant columns
selected_data = data[
    ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]
]

# Define the response variable (target) and predictors
target = "class"
predictors = [col for col in selected_data.columns if col != target]

# Handle missing values
selected_data = selected_data.fillna(selected_data.median(numeric_only=True))  # Fill numeric values with median
selected_data = selected_data.fillna(selected_data.mode().iloc[0])  # Fill categorical values with mode

# Convert target variable to integer encoding if categorical
if selected_data[target].dtype.name in ["category", "object"]:
    selected_data[target] = selected_data[target].astype("category").cat.codes

X = selected_data[predictors].values
y = selected_data[target].values

# Scale predictors
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and holdout sets
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the MLPClassifier (DNN equivalent)
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # Three layers with 128, 64, 32 neurons
    activation='relu',  # ReLU activation function
    solver='adam',  # Adam optimizer
    max_iter=200,  # Maximum number of iterations
    early_stopping=True,  # Enable early stopping for better convergence
    random_state=42
)

# Perform 10-fold cross-validation
print("Performing 10-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"  Fold {fold} in progress...")
    mlp_clone = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
    # Train on the current fold
    mlp_clone.fit(X_train[train_idx], y_train[train_idx])

    # Evaluate on validation set
    val_accuracy = mlp_clone.score(X_train[val_idx], y_train[val_idx])
    cv_accuracies.append(val_accuracy)

# Print Cross-Validation Results
cv_mean_accuracy = np.mean(cv_accuracies) * 100
cv_std_accuracy = np.std(cv_accuracies) * 100
print("\nCross-Validation Results:")
print(f"  Mean Accuracy: {cv_mean_accuracy:.2f}%")
print(f"  Accuracy Standard Deviation: {cv_std_accuracy:.2f}%")

# Train the final model on the full training set
print("\nTraining Final Model on Full Training Set...")
mlp.fit(X_train, y_train)

# Evaluate on the holdout set
y_holdout_pred = mlp.predict(X_holdout)

# Calculate performance metrics
accuracy = accuracy_score(y_holdout, y_holdout_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_holdout, y_holdout_pred, average=None  # Metrics for each class
)
weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(
    y_holdout, y_holdout_pred, average="weighted"
)

# Output Holdout Set Results
print("\nHoldout Set Performance Metrics:")
print(f"  Accuracy: {accuracy * 100:.2f}%")
print("  Class-wise Metrics:")
for i, (p, r, f) in enumerate(zip(precision, recall, f1_score)):
    print(f"    Class {i}: Precision: {p * 100:.2f}%, Recall: {r * 100:.2f}%, F1-Score: {f * 100:.2f}%")

print(f"\n  Weighted Average Precision: {weighted_precision * 100:.2f}%")
print(f"  Weighted Average Recall: {weighted_recall * 100:.2f}%")
print(f"  Weighted Average F1-Score: {weighted_f1_score * 100:.2f}%")

# Detailed classification report
print("\nDetailed Classification Report on Holdout Set:")
print(classification_report(y_holdout, y_holdout_pred))
