import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = "C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv"
data = pd.read_csv(data_path)

# Select relevant columns
selected_data = data[
    ["F flex", "F ext", "N mov", "R min", "R max", "t game", "P min", "P max", "P mean", "score", "class"]
]

# Columns to plot
columns_to_plot = ["R min", "R max", "P max", "P mean"]

# Plot histogram for selected predictors grouped by class
plt.figure(figsize=(10, 8))
for i, column in enumerate(columns_to_plot):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=selected_data, x=column, hue="class", kde=True, bins=20, element="step", stat="density")
    plt.title(f"Distribution of {column} by class")

plt.tight_layout()
plt.show()
