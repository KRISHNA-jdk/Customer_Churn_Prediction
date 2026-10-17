# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to discontinue their subscription using customer demographic, service, contract, and billing information.

The project demonstrates an end-to-end supervised machine learning workflow, including data cleaning, exploratory data analysis, preprocessing, model training, evaluation, and feature interpretation.

## Project Overview

Customer churn is an important business problem for telecommunications companies. Identifying customers who are more likely to leave can help businesses understand customer behavior and make informed retention decisions.

This project uses the IBM Telco Customer Churn dataset to build and compare two binary classification models:

- Logistic Regression
- Decision Tree Classifier

The models are evaluated using multiple classification metrics rather than accuracy alone.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Google Colab / Jupyter Notebook

## Dataset

The project uses the IBM Telco Customer Churn dataset.

**Dataset source:**  
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The dataset contains 7,043 customer records and 21 original columns, including customer account information, subscribed services, billing details, and churn status.

The target variable is `Churn`:

- `No` — Customer did not churn.
- `Yes` — Customer churned.

The `customerID` column is excluded from model training because it is an identifier rather than a meaningful predictive feature.

## Project Workflow

### 1. Data Cleaning

- Inspected the dataset structure and data types.
- Converted `TotalCharges` to a numeric data type.
- Handled invalid or missing values through preprocessing pipelines.

### 2. Exploratory Data Analysis

- Examined the distribution of churn and non-churn customers.
- Investigated relationships between churn and customer characteristics.
- Compared churn patterns across contract types and other service-related features.

### 3. Data Preprocessing

Numerical features:
- Missing values handled using median imputation.
- Features standardized using `StandardScaler`.

Categorical features:
- Missing values handled using most-frequent imputation.
- Categories encoded using `OneHotEncoder`.

The preprocessing steps are incorporated into Scikit-learn pipelines to keep training and prediction transformations consistent and help prevent data leakage.

### 4. Model Training

Two classification algorithms were trained:

**Logistic Regression**
- Used as a classification baseline.
- Estimates the probability of customer churn.

**Decision Tree Classifier**
- Uses a maximum depth of 5 to limit model complexity.
- Captures decision rules based on customer features.

The dataset was divided into training and testing sets using an 80:20 split with stratification.

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Results

Results from the current test-set evaluation:

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 80.55% | 79.84% |
| Precision | 65.72% | 63.47% |
| Recall | 55.88% | 56.68% |
| F1-score | 60.40% | 59.89% |
| ROC-AUC | 0.8419 | 0.8303 |

### Results Interpretation

Logistic Regression achieved higher accuracy, precision, F1-score, and ROC-AUC in this experiment.

The Decision Tree achieved slightly higher recall, identifying a marginally larger proportion of customers who actually churned.

The results demonstrate the trade-off between different evaluation metrics. The preferred model depends on the business objective and the relative costs of missed churners and unnecessary retention outreach.

These results come from a single train-test split and should not be interpreted as a guarantee of performance on future data.

## Feature Analysis

Feature coefficients from Logistic Regression and feature importance scores from the Decision Tree were examined to understand which variables contributed to model predictions.

Important predictive features included:

- Contract type
- Internet service type
- Customer tenure
- Total charges
- Monthly charges
- Payment method

These relationships indicate predictive associations in the dataset; they do not establish that individual features cause customer churn.

## Classification Threshold Analysis

The project also explores how changing the classification threshold affects precision and recall.

A lower threshold can identify more potential churners but may also increase false positives. A higher threshold generally reduces positive predictions, potentially improving precision while missing more actual churners.

Threshold selection should depend on business requirements and should be performed using validation data rather than repeatedly optimizing against the final test set.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/KRISHNA-jdk/Customer_Churn_Prediction.git
```

Replace the repository URL if your GitHub repository uses a different name.

### 2. Navigate to the project

```bash
cd Customer_Churn_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

Run Jupyter Notebook:

```bash
jupyter notebook
```

Alternatively, upload `Customer_Churn_Prediction.ipynb` to Google Colab and execute the notebook cells.

The notebook downloads the dataset directly from its source.

## Limitations and Future Improvements

- Evaluate model stability using stratified cross-validation.
- Perform systematic hyperparameter tuning.
- Select classification thresholds using validation data and business costs.
- Evaluate the final model on an untouched test set.
- Explore additional classification algorithms.
- Investigate model calibration and deployment options.

## Key Learnings

- Building an end-to-end supervised machine learning workflow.
- Handling missing values and mixed numerical and categorical data.
- Using Scikit-learn pipelines for preprocessing and model training.
- Comparing classification models using multiple evaluation metrics.
- Interpreting model coefficients and feature importance.
- Understanding precision-recall trade-offs in an imbalanced classification problem.

## Author

**Krishna Rawat**

- GitHub: https://github.com/KRISHNA-jdk
- LinkedIn: https://www.linkedin.com/in/krishna-rawat-98b18432a

---

This project is intended for educational purposes and does not represent a production-ready customer retention system.
