# Task 1: Student Score Prediction (Polynomial Regression)

## 📌 Objective
Build a regression model to predict **student exam scores (`Exam_Score`)** based on factors such as **study hours, sleep, participation, and other features**.  
The task also compares **Linear Regression** with **Polynomial Regression** of higher degrees.

---

## 🛠 Steps

### 1. Data Preparation
- Loaded dataset: `StudentPerformanceFactors.csv`  
- Split into **train (80%)** and **test (20%)**.  
- Identified numerical and categorical features.  

### 2. Preprocessing Pipeline
- **Numerical features**: Missing values replaced with **median**, standardized with `StandardScaler`.  
- **Categorical features**: Missing values replaced with **mode**, encoded using `OneHotEncoder`.  
- Combined using `ColumnTransformer`.  

### 3. Modeling
- Built a pipeline with:
  - Preprocessor  
  - PolynomialFeatures (degree = 1, 2, 3)  
  - Linear Regression  

- Trained models on **training data**.  
- Evaluated models on **validation data** using:
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
  - R² score  

### 4. Model Saving
- Each model (degree 1, 2, 3) saved separately as:
  - `best_model_degree_1.pkl`
  - `best_model_degree_2.pkl`
  - `best_model_degree_3.pkl`

### 5. Evaluation on Test Data
- The **best model (degree 1)** was loaded.  
- Predictions made on **test data**.  
- Final evaluation metrics reported.  

### 6. Visualization
- Scatter plots of **training and validation data**.  
- Prediction curves for **degree 1 (blue)**, **degree 2 (green)**, **degree 3 (red)**.  
- Compared fits visually to check underfitting/overfitting.  

---

## 📊 Results
- **Degree 1 (Linear Regression)** performed best in validation and test sets.  
- Higher-degree polynomial models (2, 3) showed signs of **overfitting**.  
- Final evaluation on test data included **MAE, RMSE, R²**.  

---

## 📂 Files
- `Task1_StudentScorePrediction.ipynb` → Full notebook with code + analysis  
- `README.md` → This document  
- `best_model_degree_1.pkl` → Best saved model (Linear Regression)  
- `best_model_degree_2.pkl` → Saved model with polynomial degree 2  
- `best_model_degree_3.pkl` → Saved model with polynomial degree 3  

---

## 🧰 Tools & Libraries
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## 🚀 Next Steps
- Try **regularization techniques** (Ridge, Lasso) to improve model stability.  
- Experiment with **feature engineering** (e.g., interaction terms, removing irrelevant features).  
- Perform **hyperparameter tuning** for better performance.  
