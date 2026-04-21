# 💳 Credit Card Fraud Detection System
### Machine Learning Project | Python | Scikit-Learn | K-Fold Cross Validation

---

## 📌 Project Overview
This project detects fraudulent credit card transactions using supervised machine learning models. The dataset contains **284,807 real transactions** from European cardholders, of which only **0.17% are fraudulent** — making it a highly imbalanced classification problem.

All models are evaluated using **Stratified K-Fold Cross Validation (cv=5)** for reliable and unbiased performance measurement.

---

## 📂 Dataset
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Size:** 284,807 transactions | 31 features
- **Target:** `Class` → 0 = Legitimate, 1 = Fraud

> ⚠️ Download `creditcard.csv` from Kaggle and place it in the same folder as the notebook before running.

---

## 🛠️ Technologies Used
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and preprocessing |
| Scikit-Learn | ML models, K-Fold, GridSearch, metrics |
| Matplotlib & Seaborn | Data visualization |
| Jupyter Notebook | Development environment |

---

## 🔄 Project Workflow

```
Dataset (creditcard.csv)
        ↓
Data Cleaning & EDA
        ↓
Feature Scaling + Class Imbalance Handling (Undersampling)
        ↓
Model Training with Stratified K-Fold (cv=5)
        ↓
Evaluation: Accuracy, Confusion Matrix, ROC-AUC
        ↓
Hyperparameter Tuning (GridSearchCV)
        ↓
Fraud Prediction System (predict_fraud)
```

---

## 📊 Models Used
| Model | Type |
|-------|------|
| 🔵 Logistic Regression | Linear baseline |
| 🟠 Decision Tree | Rule-based non-linear |
| 🟢 Random Forest | Ensemble — Best Model ✅ |

---

## 📈 Key Results
| Model | K-Fold Accuracy |
|-------|----------------|
| Logistic Regression | ~93–95% |
| Decision Tree | ~92–95% |
| **Random Forest** | **~95–97% ✅** |

---

## 🔍 Project Sections
1. Introduction & Problem Statement
2. K-Fold Cross Validation Theory
3. Library Imports
4. Dataset Loading & Exploration
5. Exploratory Data Analysis (EDA)
   - Fraud vs Legitimate Distribution
   - Transaction Amount Analysis
   - Time-based Fraud Patterns
   - PCA Feature Distributions
   - Correlation Heatmap
6. Data Preprocessing & Class Imbalance Handling
7. Model Theory (LR, DT, RF)
8. K-Fold Evaluation — All Models
9. Model Accuracy Comparison (Bar Chart)
10. Per-Fold Distribution (Box Plot)
11. Confusion Matrices — All Models
12. Hyperparameter Tuning (GridSearchCV)
13. ROC Curve Comparison
14. Feature Importance (Random Forest)
15. Fraud Prediction System
16. Business Insights & Recommendations
17. Final Summary

---

## 💳 Fraud Prediction System
A `predict_fraud()` module is implemented that:
- Takes new transaction data as input
- Applies the trained Best Random Forest model
- Outputs **fraud probability score**
- Returns **FRAUDULENT** 🚨 or **LEGITIMATE** ✅ classification

```python
predict_fraud(transaction_data, best_rf, scaler, feature_names)
```

---

## 🚀 How to Run
1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection-ml.git
```
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project folder
3. Install required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
4. Open and run the notebook
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## 📁 Repository Structure
```
credit-card-fraud-detection-ml/
│
├── Credit_Card_Fraud_Detection.ipynb   ← Main notebook
├── README.md                           ← Project documentation
└── creditcard.csv                      ← Dataset (download from Kaggle)
```

---

## 👤 Author
**Hamza Asif**  
BS Artificial Intelligence — DUET University  
