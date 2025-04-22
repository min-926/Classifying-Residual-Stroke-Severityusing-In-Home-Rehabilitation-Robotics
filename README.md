# Stroke Severity Classification Using In-Home Robotic Rehabilitation Data

## Overview

This project applies supervised machine learning to classify the severity of motor impairment in stroke survivors using kinematic and pressure data collected from in-home, robot-assisted therapy sessions. By leveraging Motus Nova’s Motus Hand and Foot devices, we analyze time-series sensor data and extract key features to predict range of motion (ROM) classes: **No**, **Low**, and **High**.

## Project Objectives

- Classify stroke severity using data collected from at-home robotic therapy devices.
- Compare the performance of various classification models including:
  - Multinomial Logistic Regression (MLR)
  - Random Forest (RF)
  - LightGBM (LGBM)
  - Support Vector Machines (SVM)
- Use dimensionality reduction techniques (PCA, LDA, t-SNE) to enhance model performance and interpretability.
- Identify key biomechanical features contributing to stroke recovery classification.

## Dataset

- **Source**: Motus Nova in-home therapy sessions
- **Participants**: 33 stroke survivors
- **Devices Used**: Motus Hand & Foot
- **Data Type**: Time-series data converted to game-level summary statistics
- **Features**:
  - Range of Motion (Rmin, Rmax)
  - Pressure metrics (Pmax, Pmean)
  - Force metrics (Fflex, Fext)
  - Movement count, game time, scores
- **Labels**: Clinician-classified ROM levels (No, Low, High)

## Data Processing

- Time-series data was cleaned, imputed, and summarized.
- Outlier smoothing and normalization were applied.
- Dimensionality reduction techniques (PCA, LDA, t-SNE) were used for visualization and feature selection.
- The dataset was split into an 80/20 train-test ratio.

## Models and Performance

| Model                   | Accuracy | Weighted F1 Score |
|------------------------|----------|-------------------|
| Random Forest          | 99.7%    | 99.7%             |
| LightGBM               | 97.0%    | 97.0%             |
| Logistic Regression    | 61.0%    | 56.0%             |
| Support Vector Machine | 60.0%    | 55.0%             |

**Best Performing Model**: **Random Forest**, followed closely by **LightGBM**

## Key Insights

- Ensemble models (Random Forest and LightGBM) significantly outperformed traditional classifiers.
- LDA highlighted Pmax, Fflex, and game time as influential predictors.
- t-SNE revealed class overlap but also separable clusters, especially for "High" and "No" ROM categories.

## Future Work

- Test generalizability with external datasets
- Address class imbalance with advanced resampling techniques
- Improve real-time prediction capabilities for clinical deployment
- Integrate clinician feedback into model refinement

## Author

**Minjuan Meng**  
Results-driven data analyst with expertise in machine learning, statistical modeling, and healthcare analytics.  
📫 [LinkedIn](#) | 📧 [Email](#)

## Advisor

**Professor Russell Jeter, Ph.D.**  
Assistant Professor of Math & Statistical Foundations of Big Data  
Department of Mathematics and Statistics  
**Georgia State University**
