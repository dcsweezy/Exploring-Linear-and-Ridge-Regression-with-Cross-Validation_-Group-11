# SC5002_Assignment2 – Linear and Ridge Regression on Insurance Dataset

## 🧠 Project Overview
This project explores **Linear Regression** and **Ridge Regression** models to predict **medical insurance costs** based on various personal and demographic factors.  
It was developed as part of the **SC5002 Machine Learning Assignment 2**, focusing on understanding model performance, overfitting control using regularization, and evaluation via cross-validation.

---

## 📊 Dataset Description
We used the [**Medical Cost Personal Dataset**](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle, which contains 1,338 observations with the following attributes:

| Feature | Description | Type |
|----------|--------------|------|
| **age** | Age of the primary beneficiary | Numeric |
| **sex** | Gender of the beneficiary (male/female) | Categorical |
| **bmi** | Body Mass Index, a measure of body fat | Numeric |
| **children** | Number of dependents covered by insurance | Numeric |
| **smoker** | Whether the person is a smoker (yes/no) | Categorical |
| **region** | Residential area in the US (northeast, southeast, etc.) | Categorical |
| **charges** | Final medical insurance cost (target variable) | Numeric |

---

## 🧩 Objective
- To predict **insurance charges** using regression models.  
- To compare **Linear Regression** and **Ridge Regression** in terms of performance and generalization.  
- To understand the effect of **regularization parameter α** in Ridge Regression.  
- To use **cross-validation** for reliable model evaluation.

---

## ⚙️ Steps Taken

### 1. **Data Preprocessing**
- Loaded the dataset into a Pandas DataFrame.  
- Checked for missing values and data consistency.  
- Encoded categorical variables using **One-Hot Encoding**.  
- Scaled numerical features using **StandardScaler** for balanced regression input.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized data distributions using `seaborn` histograms and boxplots.  
- Observed relationships between `charges` and features like `age`, `bmi`, and `smoker`.  
- Noted that **smokers** tend to have significantly higher charges.

### 3. **Model Implementation**
- Split the data into **80% training** and **20% testing** sets.  
- Trained two regression models:
  - **Linear Regression**
  - **Ridge Regression** (with Grid Search over α values)

### 4. **Model Evaluation**
- Evaluated using:
  - Mean Squared Error (MSE)
  - R² Score
  - Cross-validation (5-fold)
- Compared performance across models and different α values.

### 5. **Hyperparameter Tuning**
- Used **GridSearchCV** to identify the optimal α for Ridge Regression.  
- Chose the α with the highest average cross-validation R².

### 6. **Visualization**
- Plotted predicted vs. actual values.  
- Visualized the effect of α on Ridge model performance.

---

## 💡 Key Insights

- **Smoker status** is the strongest predictor of higher medical costs.
- **Ridge Regression** slightly outperforms Linear Regression on test data, showing better generalization.
- Increasing α reduces model variance but may slightly increase bias — a trade-off evident in the results.
- Proper feature scaling is crucial for regularized models like Ridge.

---

## 📈 Results Summary

| Model | R² (Train) | R² (Test) | MSE (Test) |
|--------|-------------|------------|-------------|
| Linear Regression | 0.75 | 0.70 | Moderate |
| Ridge Regression (α=10) | 0.74 | 0.72 | Slightly Lower |

*(Values are indicative : exact numbers depend on training split.)*

---

## 🧰 Requirements
To run the notebook, install dependencies:
```bash
conda create -n SC5002_Assignment2 python=3.11
conda activate SC5002_Assignment2
conda install numpy pandas scikit-learn matplotlib seaborn jupyter
