
data <- read.csv("C:/mathmatic and statistics/work/GRA/fall 2024/data/user_data_with_label.csv")
head(data)
# smoth outliers of raw data 

# Define a function to smooth outliers for a numeric vector
smooth_outliers <- function(x) {
   if (is.numeric(x)) {
      # Calculate the 1st and 99th percentiles
      lower_bound <- quantile(x, 0.01, na.rm = TRUE)
      upper_bound <- quantile(x, 0.99, na.rm = TRUE)
      
      # Replace values outside bounds with the respective percentile values
      x[x < lower_bound] <- lower_bound
      x[x > upper_bound] <- upper_bound
   }
   return(x)
}

# step2: remove number of movement less than 3


# step3: Correlation matrix
# Select the relevant columns for the correlation matrix
selected_columns <- data[, c("F.flex", "F.ext", "N.mov", "R.min", "R.max", "t.game", "P.min", "P.max", "P.mean", "score")]

# Compute the correlation matrix
correlation_matrix <- cor(selected_columns, use = "complete.obs")

# Print the correlation matrix
print(correlation_matrix)
library(corrplot)

# Visualize the correlation matrix

corrplot(correlation_matrix, method = "color", addCoef.col = "black", number.cex = 0.7)

# step 4 Dimensionality Reduction: PAC
library(ggplot2)
library(factoextra)
# Select the relevant numeric columns for PCA
numeric_data <- data[, c("F.flex", "F.ext", "N.mov", "R.min", "R.max", 
                         "t.game", "P.min", "P.max", "P.mean", "score")]

# Check for any missing values and remove them if necessary
numeric_data <- na.omit(numeric_data)
# Standardize the data
numeric_data_scaled <- scale(numeric_data)
# Perform PCA
pca_result <- prcomp(numeric_data_scaled, center = TRUE, scale. = TRUE)
# Check the explained variance for each principal component
summary(pca_result)
# Create a scree plot
fviz_eig(pca_result)
# Create a biplot
biplot(pca_result)
# Get the loadings (contributions) of each PC
pca_result$rotation

# Extract explained variance
explained_variance <- summary(pca_result)$importance[2,]  # Proportion of variance

# Cumulative variance
cumulative_variance <- cumsum(explained_variance)

# Create a data frame for plotting
variance_data <- data.frame(
   PC = 1:length(explained_variance),
   ExplainedVariance = explained_variance,
   CumulativeVariance = cumulative_variance
)

# Plot using ggplot2
ggplot(variance_data, aes(x = PC)) +
   geom_bar(aes(y = ExplainedVariance), stat = "identity", fill = "purple", alpha = 0.6) +
   geom_line(aes(y = CumulativeVariance), color = "blue", linewidth = 1.0) + 
   scale_y_continuous(labels = scales::percent) +
   labs(
      title = "PAC Graph: Explained Variance of Principal Components",
      x = "Principal Component Index",
      y = "Explained Variance Ratio"
   ) +
   theme_minimal() +
   theme(legend.position = "top") +
   scale_x_continuous(breaks = 1:length(explained_variance)) +
   scale_color_manual(name = "Variance", values = c("purple", "blue")) +
   guides(fill = "none")


#step5 LGB model
























