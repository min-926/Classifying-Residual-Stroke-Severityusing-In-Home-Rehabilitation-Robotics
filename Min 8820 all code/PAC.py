from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

# Select predictor variables (excluding 'class')
predictor_cols = ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score"]
X = data[predictor_cols]

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_filled = imputer.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(predictor_cols) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by Principal Components")
plt.grid()
plt.show()
