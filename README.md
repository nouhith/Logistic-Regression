# 🧪 Logistic Regression on Breast Cancer Dataset

This repository presents a comprehensive implementation of **Logistic Regression** on a **binary classification dataset** — the Breast Cancer dataset. The goal is to classify tumors as either **benign** or **malignant** using features derived from digitized images of breast masses.

This notebook walks through all essential steps of a binary classification pipeline — from data preprocessing to model evaluation and interpretation.

---

## 📁 File Structure

- **Breast_cancer.csv**:  
  The dataset containing labeled breast cancer diagnostic data.
  
- **Logistic Regression.ipynb**:  
  Jupyter notebook that includes all code, explanations, evaluation, and visualizations.

---

## 📊 Classification Workflow Overview

### 1. 📂 Dataset Selection

- Using the **Breast Cancer dataset** which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- The **target variable** is binary:  
  `0 = Benign`, `1 = Malignant`.

---

### 2. 🔀 Train/Test Split & Feature Scaling

- Dataset is split into **training** and **testing** sets using `train_test_split()`.
- Standardization is applied using `StandardScaler` to ensure features are on the same scale.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

### 3. 🧠 Model Training with Logistic Regression

- Logistic Regression is implemented using `sklearn.linear_model.LogisticRegression`.
- Fit on the training set and used to predict on the test set.

```python
from sklearn.linear_model import LogisticRegression
```

---

### 4. 📉 Evaluation Metrics

Evaluated the model using various metrics:

- **Confusion Matrix**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC Curve & AUC Score**

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
```

Visualization includes:
- ROC Curve with AUC
- Heatmap for Confusion Matrix

---

### 5. 🎯 Threshold Tuning & Sigmoid Explanation

- The logistic regression outputs **probabilities** using the **sigmoid function**:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where `z` is the linear combination of input features.  
This probability is converted into a class label using a threshold (default is `0.5`).

- We explore the effect of **changing the threshold** to balance between **precision** and **recall**, especially useful in **medical diagnoses** where false negatives are critical.

```python
y_proba = model.predict_proba(X_test)[:, 1]
custom_threshold = 0.3
y_pred_custom = (y_proba > custom_threshold).astype(int)
```

---

## 🔧 Libraries & Tools Used

- **Python 3.x**
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **Matplotlib** & **Seaborn** – data visualization
- **Scikit-learn** – model building, evaluation, and preprocessing

---

## 🚀 Getting Started

1. Clone this repository:

```bash
git clone https://github.com/your-username/logistic-regression-breast-cancer.git
cd logistic-regression-breast-cancer
```

2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Launch the Jupyter Notebook:

```bash
jupyter notebook Logistic_Regression_BreastCancer.ipynb
```

---

## 🧠 Key Insights

- Logistic regression is effective for binary classification problems like medical diagnostics.
- Evaluation metrics beyond accuracy (like precision and recall) are critical in imbalanced datasets.
- Tuning the classification threshold can significantly improve outcomes based on domain needs (e.g., minimizing false negatives).

---

## 📌 Note

This project focuses on **model training, evaluation, and interpretation**. It serves as a robust base for applying logistic regression in real-world binary classification tasks.

---

## 🧑‍💻 Author

Created by **Nouhith**  
Feel free to explore, fork, and contribute to the project!
