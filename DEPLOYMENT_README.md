# Startup Success Prediction Project Report



<p align="center">
  <!-- Langages et Librairies -->
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2.2.2-17BEBB?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white" />

  <!-- Machine Learning -->
  <img src="https://img.shields.io/badge/Scikit--Learn-1.5.1-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-4.2.0-00B0F0?logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/ExtraTrees-ML-lightgrey?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SVM-ML-6F42C1?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Logistic_Regression-ML-007ACC?logo=scikit-learn&logoColor=white" />

  <!-- Preprocessing & Feature Engineering -->
  <img src="https://img.shields.io/badge/OneHotEncoding-✔️-yellow" />
  <img src="https://img.shields.io/badge/Scaling-Normalization-FF5733" />
  <img src="https://img.shields.io/badge/Imputation-✔️-C70039" />
  <img src="https://img.shields.io/badge/SMOTE-✔️-00FF00" />
  <img src="https://img.shields.io/badge/SMOTEENN-✔️-32CD32" />
  <img src="https://img.shields.io/badge/Borderline_SMOTE-✔️-3CB371" />
  <img src="https://img.shields.io/badge/TomekLinks-✔️-2E8B57" />

  <!-- Hyperparameter Tuning -->
  <img src="https://img.shields.io/badge/GridSearch-✔️-8A2BE2" />
  <img src="https://img.shields.io/badge/RandomSearch-✔️-9370DB" />
  <img src="https://img.shields.io/badge/BayesianOptimization-Optuna-FF1493" />

  <!-- Visualisation -->
  <img src="https://img.shields.io/badge/Matplotlib-✔️-11557C" />
  <img src="https://img.shields.io/badge/Seaborn-✔️-4060A0" />
  <img src="https://img.shields.io/badge/SHAP-✔️-FF8C00" />
  <img src="https://img.shields.io/badge/LIME-✔️-FFD700" />

  <!-- Backend & Deployment -->
  <img src="https://img.shields.io/badge/Flask-2.3.2-000000?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Gunicorn-22.0.0-499848?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Render-Cloud-FF5C5C?logo=render&logoColor=white" />

  <!-- Frontend -->
  <img src="https://img.shields.io/badge/HTML/CSS/JS-Frontend-F7DF1E?logo=javascript&logoColor=black" />

  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 👥 Team Members

<p align="center">
  <b>Farhan Akhtar</b> &nbsp; | &nbsp;
  <b>Aniket Sanjay Gazalwar</b> &nbsp; | &nbsp;
  <b>Souleymane Diallo</b> &nbsp; | &nbsp;
  <b>Hinimdou Morsia Guitdam</b> &nbsp; | &nbsp;
  <b>Samridh Shrotriya</b>
</p>


## Table of Contents
1. [Introduction](#1-introduction)
   - 1.1 [Project Background](#11-project-background)
   - 1.2 [Objectives](#12-objectives)

2. [Project Overview](#2-project-overview)
   - 2.1 [Dataset Description](#21-dataset-description)
   - 2.2 [Features and Target Variable](#22-features-and-target-variable)

3. [Challenges and Complexity](#3-challenges-and-complexity)
   - 3.1 [Data Quality and Missing Values](#31-data-quality-and-missing-values)
   - 3.2 [Heterogeneity and Variability of Startups](#32-heterogeneity-and-variability-of-startups)
   - 3.3 [Class Imbalance and Correlations](#33-class-imbalance-and-correlations)
   - 3.4 [Technologies Used](#34-technologies-used)

4. [Challenge Management and Solutions](#4-challenge-management-and-solutions)
   - 4.1 [Data Cleaning and Preprocessing](#41-data-cleaning-and-preprocessing)
   - 4.2 [Feature Engineering](#42-feature-engineering)
   - 4.3 [Labeling and Encoding](#43-labeling-and-encoding)
   - 4.4 [Technologies Used](#44-technologies-used)

5. [Approach and Methodology](#5-approach-and-methodology)
   - 5.1 [Machine Learning Models](#51-machine-learning-models)
   - 5.2 [Pipeline Automation](#52-pipeline-automation)
   - 5.3 [Technologies Used](#53-technologies-used)

6. [Model Building and Deployment](#6-model-building-and-deployment)
   - 6.1 [Preprocessing Pipeline](#61-preprocessing-pipeline)
   - 6.2 [Target Variable Generation (`Active_Status`)](#62-target-variable-generation-active_status)
   - 6.3 [API Integration for Backend](#63-api-integration-for-backend)
   - 6.4 [Technologies Used](#64-technologies-used)

7. [Results and Discussion](#7-results-and-discussion)
   - 7.1 [Cleaned Dataset](#71-cleaned-dataset)
   - 7.2 [Model Performance](#72-model-performance)
   - 7.3 [Limitations](#73-limitations)
   - 7.4 [Technologies Used](#74-technologies-used)

8. [Conclusion](#8-conclusion)
   - 8.1 [Summary of Achievements](#81-summary-of-achievements)

9. [References](#9-references)

---

## List of Abbreviations
| **Abbreviation** | **Meaning**                           |
|------------------|---------------------------------------|
| XGB              | eXtreme Gradient Boosting             |
| LGBM             | Light Gradient Boosting Machine       |
| RF               | Random Forest                          |
| SVM              | Support Vector Machine                 |
| LR               | Logistic Regression                    |
| ETC              | Extra Trees Classifier                 |
| SMOTE            | Synthetic Minority Oversampling Technique |
| SMOTEENN         | Combination of SMOTE + Edited Nearest Neighbors |
| Borderline-SMOTE | Borderline Synthetic Minority Oversampling Technique |
| API              | Application Programming Interface     |


## 1. Introduction

### 1.1 Project Background

This project, carried out by our team between **August 1, 2025, and October 25, 2025**, as part of a **Data Science internship at Technocolabs**, aims to predict the future outcomes of startups based on their historical data.  

The dataset comes from Crunchbase and provides detailed information on startups, their funding rounds, acquisitions, IPOs, as well as industry sector, location, and other metrics related to startup success.  

It is important to note that any dataset with high diversity can pose challenges for modeling. In this project, we worked with a **highly imbalanced and heterogeneous dataset**, where the different classes (Operating, IPO, Acquired, Closed) are not equally represented. This posed a real challenge for building a robust predictive model.

---

### 1.2 Objectives

The main objective of this project is to **predict the future outcomes of startups** by classifying each company into one of the following categories:

- **Operating**: the startup is still active and generating revenue.  
- **IPO**: the startup has gone public through an Initial Public Offering.  
- **Acquired**: the startup has been acquired by a larger company.  
- **Closed**: the startup has shut down or gone bankrupt.  

To achieve this goal, we built a complete pipeline including:  
- Data collection and cleaning  
- Preprocessing and feature engineering  
- Model training and evaluation  
- Final integration into an operational pipeline  

This project demonstrates the ability to handle complex and imbalanced data while providing reliable predictions that can be leveraged by investors and stakeholders.

---

## 2-project-overview
### 2.1 Dataset Description

The project uses **Crunchbase data**, which contains a wealth of information on startups, their funding rounds, acquisitions, IPOs, and other metrics related to startup success.  

The dataset can be accessed via the **Crunchbase API** or downloaded from the provided GitHub link. It includes information such as the company name, industry, funding, founding year, number of employees, and other indicators of success.  

#### 📘 Key Variable Dictionary

| Variable Name             | Definition |
|---------------------------|------------|
| `id`                      | Unique identifier for each startup record. |
| `entity_type`             | Type of entity (e.g., company, investor, school, etc.). |
| `entity_id`               | Unique internal ID associated with the entity. |
| `parent_id`               | ID of the parent company if the startup is a subsidiary. |
| `name`                    | Official name of the startup. |
| `normalized_name`         | Standardized or lowercase version of the startup name. |
| `permalink`               | URL-friendly version of the startup name. |
| `category_code`           | Industry or sector of the startup (e.g., fintech, edtech). |
| `status`                  | Current status: Operating, Acquired, IPO, Closed. |
| `founded_at`              | Date the startup was founded. |
| `closed_at`               | Date the startup shut down (if applicable). |
| `domain`                  | Startup’s website domain. |
| `homepage_url`            | URL of the homepage. |
| `twitter_username`        | Startup’s Twitter handle. |
| `logo_url` / `logo_width` / `logo_height` | URL and dimensions of the logo. |
| `short_description`       | Brief description or tagline. |
| `description`             | Longer description of the mission or business. |
| `overview`                | Comprehensive overview of operations and market. |
| `tag_list`                | Comma-separated list of keywords or tags. |
| `country_code` / `state_code` / `city` / `region` | Startup’s geographic location. |
| `first_investment_at` / `last_investment_at` | Dates of first and most recent investments. |
| `investment_rounds`       | Number of investment rounds. |
| `invested_companies`      | Number of companies this startup has invested in. |
| `funding_rounds` / `funding_total_usd` | Total number of funding rounds and total amount raised. |
| `first_milestone_at` / `last_milestone_at` | Dates of first and most recent milestones. |
| `milestones`              | Number of key milestones achieved. |
| `relationships`           | Number of known professional relationships. |
| `created_by` / `created_at` / `updated_at` | Record creation and update information. |
| `lat` / `lng`             | Geographic coordinates of the startup. |
| `ROI`                     | Return on Investment indicator (if available). |

This rich and detailed dataset captures multiple dimensions of startup activity, which is essential for building a robust predictive model.

---

### 2.2 Features and Target Variable

In this project, we use various **features** to predict the future status of startups. Features are the explanatory variables from the Crunchbase dataset that describe each startup. They include:

- **General Information**: `name`, `normalized_name`, `category_code` (industry), `founded_at` (founding year), `country_code`, `state_code`, `city`  
- **Financial and Funding Data**: `funding_rounds` (number of funding rounds), `funding_total_usd` (total amount raised), `first_funding_at`, `last_funding_at`  
- **Milestones and Relationships**: `milestones` (number of milestones achieved), `first_milestone_at`, `last_milestone_at`, `relationships` (number of professional relationships)  
- **Other Information**: `number_of_employees`, `domain`, `homepage_url`, `twitter_username`, `ROI`  

The **target variable** is the **future status of the startup**, which can take one of four values:

- `Operating`: the startup is still active and generating revenue  
- `IPO`: the startup has gone public through an Initial Public Offering  
- `Acquired`: the startup has been acquired by a larger company  
- `Closed`: the startup has shut down or gone bankrupt  

These features allow the model to capture multiple aspects of a startup’s success and sustainability, enabling accurate predictions of its future status.  

> As you go through this README, we will also highlight **new observations and insights** identified during the project analysis, to track progress and key points at each stage.

---


## 3. Challenges and Complexity

### 3.1 Data Quality and Missing Values

We first faced **missing value issues** in several columns:  
- Columns with over 96% missing values were **dropped**.  
- Instances with missing values in `status`, `country_code`, `category_code`, or `founded_at` were **removed**.  
- For `active_days`, a **Shapiro-Wilk test** was conducted to check normality:  
  - If data was normally distributed, missing values were filled with the **mean**.  
  - Otherwise, missing values were filled with the **median**.

### 3.2 Heterogeneity and Variability of Startups

Startups vary widely in **sector, size, age, location, and funding history**.  
To handle this variability, we:  
- Normalized and scaled numerical features.  
- Encoded categorical variables using **one-hot encoding** or embeddings.  
- Analyzed sector-specific trends to adapt the analysis.

### 3.3 Class Imbalance and Correlations

The target variable (`status`) is **highly imbalanced**: most startups are still operating, while IPOs and acquisitions are rare.  
Our approach included:  
- Resampling techniques: NearMiss, Tomek Links, SMOTE, SMOTE + Tomek, SMOTEENN, SMOTE Bagging, SMOTE Boosting.  
- Class weight adjustments during model training.  
- Correlation analysis to remove redundant features.

### 3.4 Technologies Used

The tools and technologies applied to tackle these challenges included:  
- **Python** for data manipulation and preprocessing  
- **Pandas / NumPy** for dataset handling and missing value management  
- **Scikit-learn** for modeling and handling class imbalance  
- **Matplotlib / Seaborn** for visualizing distributions, correlations, and imbalances  
- **Jupyter Notebook / Google Colab / VS Code** for documenting, experimenting, and interactive development  

These solutions allowed us to handle noisy, heterogeneous, and imbalanced data effectively, improving the robustness of the predictive model.

---

## 4. Challenge Management and Solutions 
### 4.1 Data Cleaning and Preprocessing 

Data cleaning followed several steps to improve dataset quality:

1. **Removal of unnecessary or redundant columns**:  
   - Columns with excessive missing values (`parent_id` with over 100% missing, as well as those >98% null).  
   - Too granular or redundant columns: `region`, `city`, `state_code`, `id`, `Unnamed:0.1`, `entity_type`, `entity_id`, `created_by`, `created_at`, `updated_at`.  
   - Columns irrelevant for prediction: `domain`, `homepage_url`, `twitter_username`, `logo_url`, `logo_width`, `logo_height`, `short_description`, `description`, `overview`, `tag_list`, `name`, `normalized_name`, `permalink`, `invested_companies`.  

2. **Removal of duplicates and critical missing values**:  
   - Instances missing values in `status`, `country_code`, `category_code`, or `founded_at`.  

3. **Outlier handling**:  
   - **IQR method** for symmetric columns (`funding_total_usd`, `funding_rounds`).  
   - **log1 transformation** for asymmetric columns.  
   - Hybrid **IQR + log1** approach for better treatment of specific columns.

4. **Removal of contradictory data**:  
   - `funding_rounds > 0` but `funding_total_usd = 0`.  
   - Negative `active_days` (closure before founding), replaced with NaN.

5. **Date conversion**:  
   - Converted to datetime: `founded_at`, `closed_at`, `first_funding_at`, `last_funding_at`, `first_milestone_at`, `last_milestone_at`.  
   - Creation of new variables: `active_days`, `isClosed` (from `closed_at` and `status`).

---

### 4.2 Feature Engineering

1. **Categorization and grouping**:  
   - `category_code` and `country_code` simplified: only categories with ≥100 occurrences kept; others grouped as "other".  
   - Top 10 categories kept, rest as "other".  
   - **One-hot encoding** for categorical columns.

2. **Transforming categories into probabilities** using `value_counts(normalize=True)` to capture relative distribution.  

3. **Feature selection**:  
   - Correlation matrix and heatmap visualization to identify strong relationships between features.  
   - **SelectKBest** to choose the most explanatory variables.  
   - **ExtraTreesClassifier** to evaluate feature importance (Top 30 displayed in horizontal bar chart).  
   - Removal of redundant or strongly correlated columns (`last_funding_days`, `last_milestone_at`).

4. **Normalization and PCA**:  
   - PCA applied to reduce dimensionality and visualize explained variance.

---

### 4.3 Labeling and Encoding 

- **Binary target variable**:  
  - 1 if `status` = `Operating` or `IPO`  
  - 0 if `status` = `Acquired` or `Closed`  
- **Multiclass target variable**: 0, 1, 2, 3 corresponding to the four statuses.  
- **Categorical column encoding**:  
  - Main categorical columns like `category_code`, `country_code` were transformed into **numerical variables** via one-hot encoding.  
  - Several other categorical columns were also converted to **numeric** to be usable by machine learning models.  
- Dates converted to numeric format to allow calculations (e.g., `active_days`).

This step ensures all features are ready for machine learning, guaranteeing compatibility with classification algorithms.

---

### 4.4 Technologies Used 

Tools and technologies used in this section include:  

- **Python**: data manipulation and model development  
- **Pandas / NumPy**: dataset handling, missing values, and outlier management  
- **Scikit-learn**: feature selection, PCA, classification, and imbalance handling  
- **Matplotlib / Seaborn**: visualization of distributions, correlations, and feature importance  
- **Jupyter Notebook / Google Colab / VS Code**: interactive experimentation and documentation  
- **Crunchbase API**: startup data retrieval  

These steps allowed us to clean, transform, and select data efficiently to build a robust and reliable predictive model.


---
---
## 5. Approach and Methodology

### 5.1 Machine Learning Models

#### Primary Setup

- **Multiclass classification** → `Extra Trees Classifier` + `SMOTEENN`
- **Binary classification** → `LightGBM` + `Borderline-SMOTE`

The initial setup aimed to handle class imbalance and capture complex nonlinear patterns efficiently.  
Both models were trained and evaluated on the same preprocessed dataset to ensure consistent benchmarking.

#### Alternative Setup (for comparison)

- **Multiclass classification** → `SVM (One-vs-One)` + `SMOTEENN`
- **Binary classification** → `Logistic Regression` + `TomekLinks`

The alternative setup was used to validate performance consistency and explore simpler, interpretable baselines.

---

#### **Side-by-Side Comparison**

| Model | Threshold | ROC-AUC | PR-AUC | Macro-F1 | Class0_Precision | Class0_Recall | Class0_F1 | Class1_Precision | Class1_Recall | Class1_F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| XGB (SMOTE tuned) | 0.30 | 0.7649 | 0.9791 | 0.5967 | 0.2025 | 0.3840 | 0.2652 | 0.9571 | 0.9009 | 0.9281 |
| XGB (SMOTE baseline) | 0.05 | 0.7271 | 0.9746 | 0.5763 | 0.1952 | 0.2235 | 0.2084 | 0.9486 | 0.9396 | 0.9441 |
| LGBM (current) | 0.55 | 0.7307 | 0.9747 | 0.5743 | 0.1804 | 0.2577 | 0.2122 | 0.9499 | 0.9232 | 0.9364 |
| RF (SMOTE) | 0.47 | 0.6868 | 0.9665 | 0.5625 | 0.1671 | 0.2082 | 0.1854 | 0.9472 | 0.9320 | 0.9395 |
| LogReg (SMOTE, scaled) | 0.40 | 0.6766 | 0.9675 | 0.5509 | 0.1434 | 0.2065 | 0.1692 | 0.9464 | 0.9191 | 0.9326 |

**Key Takeaways:**
- The **XGBoost (SMOTE tuned)** and **LightGBM** models achieved the highest overall balance between recall and precision.  
- The **ExtraTrees + LightGBM hybrid pipeline** demonstrated a stable **macro-F1** while preserving interpretability and robustness.

---

### 5.2 Pipeline Automation

#### **Pipeline Workflow Overview**

            ┌──────────────────────┐
            │   Input Data (X)     │
            └─────────┬────────────┘
                      │
                      ▼
    ┌──────────────────────────────────┐
    │ Binary Classifier (LGBM)         │
    │ → Predicts probability of class  │
    └────────────────┬─────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────┐
    │ Feature Augmentation             │
    │ → Add predicted binary prob (p₁) │
    └────────────────┬─────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────┐
    │ Multiclass Classifier (ExtraTrees)│
    │ → Predicts among 4 status classes │
    └────────────────┬─────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │   Final Prediction   │
            └──────────────────────┘


---


This design allows the **multiclass model** to leverage the **binary model’s probability output**, improving discrimination across overlapping startup status classes.

---

#### **Implementation Code**

---
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, roc_auc_score)
    
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from sklearn.ensemble import ExtraTreesClassifier

import numpy as np

import pandas as pd

---

# =========================
# Custom Classes
# =========================
class BinaryClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = LGBMClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class MulticlassClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = ExtraTreesClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class ProbabilityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1].reshape(-1, 1)


# =========================
# Data Preparation
# =========================
df = pd.read_csv('final.csv')

X = df.drop(['status_binary', 'status_encoded_multiclass'], axis=1)
y_binary = df['status_binary']
y_multiclass = df['status_encoded_multiclass']

X_train, X_test, y_binary_train, y_binary_test, y_multiclass_train, y_multiclass_test = train_test_split(
    X, y_binary, y_multiclass, test_size=0.2, random_state=42
)


# =========================
# Combined Pipeline
# =========================
class AugmentWithBinaryProb(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator if estimator is not None else LGBMClassifier()

    def fit(self, X, y=None, y_binary=None):
        if y_binary is None:
            raise ValueError("y_binary must be provided via fit(..., y_binary=...)")
        self.estimator_ = clone(self.estimator).fit(X, y_binary)
        return self

    def transform(self, X):
        p1 = self.estimator_.predict_proba(X)[:, 1].reshape(-1, 1)
        return np.hstack([X, p1])


combined_pipeline = Pipeline([
    ("augment", AugmentWithBinaryProb(estimator=LGBMClassifier())),
    ("multiclass_clf", ExtraTreesClassifier(random_state=42)),
])

# =========================
# Training and Evaluation
# =========================
combined_pipeline.fit(
    X_train,
    y_multiclass_train,
    augment__y_binary=y_binary_train
)

y_pred = combined_pipeline.predict(X_test)

acc  = accuracy_score(y_multiclass_test, y_pred)
f1m  = f1_score(y_multiclass_test, y_pred, average="macro")
f1w  = f1_score(y_multiclass_test, y_pred, average="weighted")
bacc = balanced_accuracy_score(y_multiclass_test, y_pred)

print(f"Accuracy           : {acc:.4f}")
print(f"Macro F1           : {f1m:.4f}")
print(f"Weighted F1        : {f1w:.4f}")
print(f"Balanced Accuracy  : {bacc:.4f}\n")
print("Classification report:")
print(classification_report(y_multiclass_test, y_pred))

---
---
### 5.3 Technologies Used

| **Category** | **Tools / Libraries** |
|---------------|------------------------|
| **Programming Language** | Python 3.11 |
| **Machine Learning Frameworks** | Scikit-learn, LightGBM, XGBoost |
| **Data Balancing** | SMOTE, Borderline-SMOTE, SMOTEENN, TomekLinks |
| **Data Handling** | Pandas, NumPy |
| **Evaluation Metrics** | Scikit-learn Metrics (Accuracy, F1, ROC-AUC, PR-AUC) |
| **Visualization & Analysis** | Matplotlib, Seaborn |
| **Automation** | Scikit-learn Pipeline |

✅ **Summary**  
This modular pipeline combines binary and multiclass learning into one coherent workflow, allowing the multiclass model to use the binary classifier’s learned probabilities.  
The design increases robustness, interpretability, and predictive accuracy.

---


### 6.1 Preprocessing Pipeline
The preprocessing pipeline transforms raw input data into features ready for modeling:

- Data cleaning: removal of missing values, redundant columns, and outliers.  
- Feature engineering: categorical encoding (one-hot, numeric transformation), date conversion (`active_days`).  
- Train-test split: 80%-20% for training and evaluation.  

---

### 6.2 Model Evaluation and Comparative Study
A comparative study was conducted to evaluate different model setups for both binary and multiclass classification tasks:

**Primary Setup**:
- Multiclass: Extra Trees Classifier + SMOTEENN  
- Binary: LightGBM + Borderline-SMOTE  

**Alternative Setup**:
- Multiclass: SVM (One-vs-One) + SMOTEENN  
- Binary: Logistic Regression + TomekLinks  

**Observations from Comparative Analysis**:
- **Binary Classification**: LightGBM consistently showed higher ROC-AUC and PR-AUC scores compared to Logistic Regression, especially on the minority class.  
- **Multiclass Classification**: Extra Trees Classifier outperformed SVM in terms of Macro-F1 and balanced accuracy, providing more stable predictions across all classes.  
- **Feature Augmentation**: Using the binary classifier's predicted probability as an additional feature improved the performance of the multiclass model.  
- **Robustness**: The primary setup demonstrated higher stability across cross-validation folds and lower variance compared to the alternative setup.  

**Conclusion**:  
The primary setup (LightGBM for binary + Extra Trees for multiclass with probability augmentation) was selected for deployment due to superior overall predictive performance, robustness, and interpretability.

---

### 6.3 Target Variable Generation (`Active_Status`)
- **Binary target**: 1 if `status` = `Operating` or `IPO`, 0 if `status` = `Acquired` or `Closed`.  
- **Multiclass target**: 4 classes corresponding to all status values.  

---

### 6.4 API Integration for Backend
- Flask API serves `/predict` (JSON input) and `/predict_csv` (CSV batch uploads).  
- Model (`final_model.pkl`) loaded with pickle; uses `AugmentWithBinaryProb`.  
- Frontend (`manual.html` and `upload.html`) sends input to API and displays predictions and confidence.  
- Handles missing columns, non-numeric features, and file validation.

---

### 6.5 Deployment Process
- Local development: Python 3.12.3, virtual environment.  
- Version control: GitHub main branch.  
- Cloud hosting: Render free tier, with Gunicorn as WSGI server.  
- Post-deployment testing: API curl requests, browser testing, CSV uploads.  

**Challenges & Resolutions**:
1. Python version mismatch → set `PYTHON_VERSION=3.12.3`.  
2. Dependency compilation errors → pinned scikit-learn==1.5.1, numpy==1.26.4.  
3. Pickle namespace issues → use `CustomUnpickler`.  
4. Frontend fetch and static files → update `script.js` and Flask routes.  
5. CSV upload errors → validate file, handle missing columns.  

---

### 6.6 Tools & Technologies Used

| Category                   | Tools / Libraries                                                                 |
|-----------------------------|----------------------------------------------------------------------------------|
| Programming Language        | Python 3.12.3                                                                    |
| Machine Learning Models     | Scikit-learn (ExtraTreesClassifier), LightGBM                                     |
| Data Balancing              | SMOTE, Borderline-SMOTE, SMOTEENN                                               |
| Data Handling               | Pandas, NumPy                                                                    |
| Evaluation Metrics          | Accuracy, F1, Balanced Accuracy, ROC-AUC, PR-AUC                                  |
| API & Backend               | Flask, flask-cors                                                                |
| Deployment & Server         | Render free tier, Gunicorn                                                       |
| Frontend                    | HTML/CSS/JS, Fetch API, FormData                                                |
| Custom Components           | `AugmentWithBinaryProb` for probability augmentation                              |

✅ **Summary**:  
The pipeline combines binary and multiclass learning into one coherent workflow. Binary probabilities augment multiclass features, improving prediction quality. The deployed Flask backend supports both JSON and CSV input, and the solution is robust, interpretable, and easy to maintain.

---
---
## 7. Results and Discussion

### 7.1 Cleaned Dataset
The final dataset used for model training consists of **71,500 balanced instances** for the binary model:
- **35,750 positives** and **35,750 negatives**.  
- For the multiclass model, the 4 statuses (`Operating`, `IPO`, `Acquired`, `Closed`) are represented according to their adjusted distribution.

The dataset contains **13 features** after cleaning and transformation:
- Categorical columns (`category_code`, `country_code`, etc.) encoded using **one-hot encoding**.  
- Date columns converted to numeric format to enable derived variables like `active_days`.  
- Class imbalance for the binary model addressed using **SMOTE / Borderline-SMOTE**.  

> The dataset is therefore fully ready for training the LightGBM and ExtraTreesClassifier models.

---

### 7.2 Model Performance
Overall, the trained models showed solid performance:

- **Binary Model (LightGBM + Borderline-SMOTE)**:
  - Balanced dataset: 71,500 instances
  - High accuracy, F1, and ROC-AUC (see section 5 for detailed metrics)
  
- **Multiclass Model (ExtraTreesClassifier + Augmentation)**:
  - Takes binary model probabilities as an additional feature
  - Macro-F1 and balanced accuracy are satisfactory
  - Combined approach improves robustness and predictive capability

A **global comparative analysis** shows that the binary → multiclass pipeline improves final prediction accuracy compared to a pure multiclass model.

---

### 7.3 Limitations
- The model relies entirely on the features available in the dataset; external or recent changes in startup data may affect predictions.  
- Models are trained on historical data and may be sensitive to **future distribution changes**.  
- Current deployment on **Render free tier**, causing initialization delay (~50 seconds) after inactivity.  
- No automated CI/CD: updating the model and backend requires manual deployment.

---

### 7.4 Technologies Used

| Category                   | Tools / Libraries                                                                 |
|-----------------------------|----------------------------------------------------------------------------------|
| Programming Language        | Python 3.12.3                                                                    |
| Machine Learning Frameworks | Scikit-learn, LightGBM, ExtraTreesClassifier                                     |
| Data Balancing              | SMOTE, Borderline-SMOTE, SMOTEENN                                               |
| Data Handling               | Pandas, NumPy                                                                    |
| Evaluation Metrics          | Scikit-learn Metrics (Accuracy, F1, ROC-AUC, PR-AUC, Balanced Accuracy)         |
| Visualization & Analysis    | Matplotlib, Seaborn                                                              |
| Deployment & Automation     | Flask, Render (cloud hosting), Scikit-learn Pipeline                             |

---
## 8. Conclusion

### 8.1 Summary of Achievements and Future Perspectives

This startup success prediction project successfully demonstrated the ability to transform complex and imbalanced data into an operational predictive system.

**✅ Technical Achievements:**
- Developed a robust preprocessing pipeline capable of handling the heterogeneity of startup data
- Implemented an innovative architecture combining binary and multiclass classification with feature augmentation
- Optimized models with XGBoost (SMOTE tuned) achieving a ROC-AUC of 0.7649 and a Macro F1 of 0.5967
- Delivered a deployed solution with a web interface for real-time predictions

**🚀 Business Value:**
- Provides a decision support tool for investors and stakeholders
- Enables identification of startups at risk of closure with higher accuracy
- Analyzes key factors influencing startup success across various sectors

**🔮 Challenges Overcome and Lessons Learned:**
Handling a deeply imbalanced dataset was the core challenge. Resampling techniques (SMOTE, Borderline-SMOTE) and feature engineering approaches successfully addressed this, highlighting the importance of adapting methodologies to real-world data specifics.

**📈 Future Perspectives:**
- Integration of real-time data via Crunchbase API
- Implementation of an automatic retraining system
- Expansion to market data and external economic indicators
- Development of SHAP-based interpretations for better model explainability

This project lays the foundation for a scalable startup analytics platform, combining technical rigor and strategic vision for the innovation ecosystem.

---

## 9. References

### Libraries and Frameworks Documentation
- Pandas Documentation: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- NumPy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- LightGBM Documentation: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
- XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

### Feature Engineering Resources
- Python Feature Engineering Cookbook: [https://books.google.co.in/books?id=2c_LDwAAQBAJ&printsec=frontcover](https://books.google.co.in/books?id=2c_LDwAAQBAJ&printsec=frontcover)
- Step-by-Step Process of Feature Engineering for Machine Learning: [https://www.analyticsvidhya.com/blog/2021/03/step-by-step-process-of-feature-engineering-for-machine-learning-algorithms-in-data-science/](https://www.analyticsvidhya.com/blog/2021/03/step-by-step-process-of-feature-engineering-for-machine-learning-algorithms-in-data-science/)
- Mutual Information Feature Selection: [https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8](https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8)
- Principal Component Analysis Explained: [https://builtin.com/data-science/step-step-explanation-principal-component-analysis](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
- Principal Component Analysis for Visualization: [https://machinelearningmastery.com/principal-component-analysis-for-visualization/](https://machinelearningmastery.com/principal-component-analysis-for-visualization/)

### Deployment Resources

#### AWS Deployment
- How to Deploy a Machine Learning Model on AWS EC2 (Analytics Vidhya): [https://www.analyticsvidhya.com/blog/2022/09/how-to-deploy-a-machine-learning-model-on-aws-ec2/](https://www.analyticsvidhya.com/blog/2022/09/how-to-deploy-a-machine-learning-model-on-aws-ec2/)
- Deploy ML Model on AWS EC2 Instance (MachineLearningPlus): [https://www.machinelearningplus.com/deployment/deploy-ml-model-aws-ec2-instance/](https://www.machinelearningplus.com/deployment/deploy-ml-model-aws-ec2-instance/)
- Build, Train, and Deploy ML Model on AWS SageMaker: [https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/](https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)

#### Flask Deployment
- Flask Deployment Tutorial (YouTube Playlist): [https://www.youtube.com/playlist?list=PLJ39kWiJXSiyAFG2W3CUPWaLhvR5CQmTd](https://www.youtube.com/playlist?list=PLJ39kWiJXSiyAFG2W3CUPWaLhvR5CQmTd)

#### Django Deployment
- Django Deployment Resources: [https://www.deploymachinelearning.com/](https://www.deploymachinelearning.com/)
- Deploy ML Model Using Django and REST API: [https://www.aionlinecourse.com/blog/deploy-machine-learning-model-using-django-and-rest-api](https://www.aionlinecourse.com/blog/deploy-machine-learning-model-using-django-and-rest-api)
- Deploying a Machine Learning Model Using Django (Medium): [https://medium.com/saarthi-ai/deploying-a-machine-learning-model-using-django-part-1-6c7de05c8d7](https://medium.com/saarthi-ai/deploying-a-machine-learning-model-using-django-part-1-6c7de05c8d7)
- Django Machine Learning Tutorial (MLQ): [https://www.mlq.ai/django-machine-learning/](https://www.mlq.ai/django-machine-learning/)

#### Streamlit and Heroku Deployment
- Streamlit & Heroku Deployment Playlist 1: [https://www.youtube.com/playlist?list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE](https://www.youtube.com/playlist?list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE)
- Streamlit & Heroku Deployment Playlist 2: [https://www.youtube.com/playlist?list=PLqYFiz7NM_SN2ZbhnbfwG4kTZ6oCh0aOM](https://www.youtube.com/playlist?list=PLqYFiz7NM_SN2ZbhnbfwG4kTZ6oCh0aOM)


## Team Member:
  - Farhan Akhtar
  - Aniket Sanjay   Gazalwar
  - Souleymane Diallo
  - Hinimdou Morsia Guitdam
  - Samridh Shrotriya

