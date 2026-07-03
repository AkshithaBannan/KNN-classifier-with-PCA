# Breast Cancer Classification: Logistic Regression, KNN & PCA Pipeline

This repository features an end-to-end Machine Learning pipeline implemented in Google Colab for classifying breast cancer tumors as malignant or benign using the **Breast Cancer Wisconsin dataset**. The codebase systematically evaluates **Logistic Regression** and **K-Nearest Neighbors (KNN)** classifiers, incorporates iterative validation tuning, and leverages **Principal Component Analysis (PCA)** for low-overhead feature dimensionality reduction.

---

## Pipeline Architecture & Workflow

The notebook processes biological tumor attributes through a structured pipeline to classify diagnosis targets:

1. **Environment Setup & Data Ingestion**: Seamless ingestion using Google Colab's file upload interface to parse the raw CSV attributes.
2. **Target Label Encoding**: Utilizes `LabelEncoder` to cast the categorical target field (`diagnosis`: 'M'/'B') into a clean binary classification array ($1/0$).
3. **Data Standardization**: Implements `StandardScaler` to remove variations in scale across features, transforming independent continuous attributes to zero mean and unit variance.
4. **Baseline Modeling Frameworks**: Trains initial configurations of Logistic Regression and $K$-Nearest Neighbors ($K=5$) classifiers.
5. **Hyperparameter Tuning Strategy**: Sweeps neighbor hyperparameters dynamically ($K \in [5, 15]$) across nested loops to measure and isolate performance trends.
6. **Dimensionality Reduction (PCA)**: Unsupervised projection onto orthogonal coordinate systems to retain max-variance eigenvectors, streamlining the feature space down to its first 8 main components.

---

## Model Performance Matrix & Optimization

The workflow monitors structural model performance under randomized 100-Fold cross-validation splits ($30\%$ testing holdouts) across both raw and PCA-reduced attribute matrices:

| Model Structure | Feature Space Condition | Cross-Validation Train Acc | Cross-Validation Test Acc |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | Full Dimensions (30 Features) | $0.99$ | **`0.97`** |
| **K-Nearest Neighbors ($K=5$)** | Full Dimensions (30 Features) | $0.98$ | **`0.96`** |
| **Logistic Regression + PCA** | Reduced Space (Top 8 PCs) | $0.98$ | **`0.97`** |
| **K-Nearest Neighbors + PCA** | Reduced Space (Top 8 PCs) | $0.97$ | **`0.96`** |

### Key Analytical Takeaways
* **Dimensionality Compression**: The PCA breakdown reveals that the first two components capture roughly $63.2\%$ of total variance ($44.3\%$ and $19.0\%$ respectively). By selecting the top 8 components, the feature footprint is slashed by nearly $73\%$ while preserving **identical downstream classification power** ($0.97$ test accuracy for Logistic Regression).
* **KNN Stability**: The systematic hyperparameter sweep across $K=5$ to $K=15$ revealed highly stable testing performance, maintaining a consistent test generalizability of $0.96$.

---

## Hyperparameter Optimization & Visualization

The pipeline evaluates the KNN model's neighbor parameter via a dual-loop cross-validation structure:

```python
avg_train_acc = []
avg_test_acc = []

for k in range(5, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    # 100-fold validation iteration block...
