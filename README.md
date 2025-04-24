# 🏦 Bank Customer Churn Prediction

A comprehensive machine learning project focused on predicting customer churn in the banking sector. This project includes detailed data preprocessing, feature engineering, model benchmarking, and hyperparameter tuning with a focus on model interpretability and accuracy.

🔗 **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/gamalosama/bank-churn)

---

## 📌 Objective

To identify customers likely to leave the bank using demographic, financial, and behavioral data. This helps the business proactively manage customer retention.

---

## 📁 Dataset Overview

- **Filename**: `Bank_churn.csv`
- **Target**: `Exited` (1 = churned, 0 = retained)
- **Features**: Age, Credit Score, Balance, Gender, Geography, Products Owned, Activity Status, Salary, etc.

---

## 🧠 Project Workflow

### 1. 🔍 Exploratory Data Analysis (EDA)
- Inspected missing data, outliers, and data types
- Visualized churn rates across categorical and numerical features
- Observed relationships using correlation heatmaps and distribution plots

### 2. 🧼 Data Preprocessing
- Dropped irrelevant fields like `CustomerId` and `Surname`
- Encoded categorical features using `LabelEncoder` and pipelines
- Standardized numerical features
- Detected and removed duplicates

### 3. 🔧 Feature Engineering
- Created polynomial features with pipeline support
- Constructed preprocessing pipelines using `ColumnTransformer` for clean integration

### 4. 🤖 Model Training & Benchmarking

Benchmarked the following classifiers using **5-fold cross-validation**:
- Logistic Regression
- Ridge Classifier
- Random Forest
- Decision Tree
- Extra Trees Classifier
- XGBoost

Each model was evaluated for mean and standard deviation of CV accuracy.

### 5. 🎯 Hyperparameter Optimization

Used `RandomizedSearchCV` to tune the **XGBoost** pipeline:
- Included preprocessing steps (like polynomial degree)
- Searched over tree depth, learning rate, estimators, and sampling parameters
- Utilized all CPU cores (`n_jobs=-1`) for parallelized tuning

### 6. ✅ Final Evaluation
- Best model selected via `best_estimator_`
- Evaluated on hold-out test set using:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-score (weighted)
- Output detailed classification report

---

## 📈 Results

- **Best Model**: XGBoost with tuned hyperparameters
- **Evaluation Metrics**:
  - Accuracy: ~🔒 (fill in your actual value)
  - F1 Score: ~🔒 (fill in your actual value)
- Important insights:
  - Age and activity level are strong churn indicators
  - Feature transformation improved model performance

---

## 🧰 Tools & Technologies

- Python (Jupyter Notebook)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost
- Pipelines, ColumnTransformer
- RandomizedSearchCV for tuning

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/gamal1osama/Bank-Customer-Churn.git
2. Navigate into the folder and install dependencies:
   ```bash
    pip install -r requirements.txt
3. Launch the notebook:
    ```bash
    jupyter notebook my_code1.ipynb
