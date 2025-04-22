import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

# Handle missing values
print("Checking for missing values...")
if data.isnull().sum().sum() > 0:
    print("Missing values detected. Handling missing values...")
    # Impute missing values
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:  # Numerical columns
            data[col].fillna(data[col].mean(), inplace=True)
        elif data[col].dtype == 'object':  # Categorical columns
            data[col].fillna(data[col].mode()[0], inplace=True)
else:
    print("No missing values detected.")

# Verify no missing values remain
if data.isnull().sum().sum() > 0:
    raise ValueError("Missing values remain after imputation. Please check the dataset.")
else:
    print("Missing values handled successfully.")

# Select relevant columns
selected_data = data[
    ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]]

# Define the response variable (target) and predictors
target = "class"
predictors = selected_data.drop(columns=[target]).columns.tolist()

# Convert target variable to integer encoding
selected_data[target] = selected_data[target].astype('category').cat.codes

# Separate numerical and categorical columns for preprocessing
numerical_cols = selected_data[predictors].select_dtypes(include=['float64', 'int64']).columns
categorical_cols = selected_data[predictors].select_dtypes(include=['object', 'category']).columns

# Scale numerical features
scaler = StandardScaler()
selected_data[numerical_cols] = scaler.fit_transform(selected_data[numerical_cols])

# Define predictors and target
X = selected_data[predictors]
y = selected_data[target]

# Split the dataset into training and validation sets (80/20 split, stratified)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Prepare the LightGBM dataset
dtrain = lgb.Dataset(data=X_train, label=y_train)
dval = lgb.Dataset(data=X_val, label=y_val, reference=dtrain)

# Define model parameters
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(y.unique()),
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 40,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1
}

# Train the model with early stopping and logging
model = lgb.train(
    params=params,
    train_set=dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    valid_names=['validation'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=10)
    ]
)

# Predict on the validation set
val_predictions = model.predict(X_val)
val_predicted_classes = np.argmax(val_predictions, axis=1)

# Evaluate performance
accuracy = accuracy_score(y_val, val_predicted_classes)
print("\nValidation Set Metrics:")
print("Accuracy (%):", accuracy * 100)
print("Confusion Matrix:\n", confusion_matrix(y_val, val_predicted_classes))
print("\nClassification Report:\n", classification_report(y_val, val_predicted_classes))

