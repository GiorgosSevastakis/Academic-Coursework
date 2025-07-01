# Applied Machine Learning – Solo Project

This folder contains the solo project from the *Applied Machine Learning* course (May 2024), involving classification, regression, and clustering tasks applied on CERN's LHC data. It demonstrates the use of both tree-based models and neural networks with model interpretability and hyperparameter optimization.

---

## Problem Summary

The project is divided into three parts:

1. **Classification**  
   Binary classification using both **XGBoost** and **TensorFlow**. Feature importance was determined using built-in scores and SHAP values, and hyperparameter tuning was performed via **Bayesian optimization** (Hyperopt). The best XGBoost model achieved a cross-validated log-loss of ~0.07, while the neural network achieved ~0.115.

2. **Regression**  
   Regression models were built using **XGBoost** and **PyTorch**. Feature selection was based on permutation importance, and models were evaluated using **Mean Absolute Error**. XGBoost outperformed PyTorch with an MAE of 0.135 vs. 0.2106, respectively.

3. **Clustering**  
   The dataset was clustered using **KMeans**, **Agglomerative Clustering**, and **DBSCAN**, based on the top 10 features selected using Laplacian scores. Evaluation methods like Calinski-Harabasz index, elbow method, and k-distance plot were used to identify the optimal number of clusters.

---

## Methods Used

- Standardization and feature selection (Laplacian scores, SHAP, permutation importance)
- Supervised learning with XGBoost, TensorFlow (Keras), and PyTorch
- Hyperparameter optimization via **Hyperopt** and random search
- Clustering analysis and evaluation using CH index, elbow method, and DBSCAN heuristics

---

## Files

- `Classification/`: Scripts for XGBoost and TensorFlow classification
- `Regression/`: Scripts for XGBoost and PyTorch regression
- `Clustering/`: KMeans, Agglomerative, and DBSCAN clustering
- `description.pdf`: Detailed explanation of the full workflow and results

---

## Author

Georgios Sevastakis  

May 2024
