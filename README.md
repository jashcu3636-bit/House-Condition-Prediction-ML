# 🏠 House Price Prediction using Machine Learning

<p align="center">

**An end-to-end Machine Learning project for predicting residential property prices using Linear Regression.**

</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-2.3.2-150458?logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-2.3.3-013243?logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-3.10.6-11557C?logo=matplotlib&logoColor=white" />

</p>

---

## 📌 Overview

This project implements a complete introductory **supervised Machine Learning workflow** for predicting house prices from structured property data.

The project starts with raw tabular data and follows a reproducible pipeline:

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Inspection & Validation
     │
     ▼
Data Preprocessing
     │
     ├── Missing Value Check
     ├── Duplicate Check
     └── Categorical Encoding
     │
     ▼
Feature / Target Separation
     │
     ▼
Train-Test Split
     │
     ├───────────────┐
     ▼               ▼
Training Set      Testing Set
     │               │
     ▼               │
Linear Regression   │
     │               │
     └───────┬───────┘
             ▼
        Predictions
             │
             ▼
     Model Evaluation
             │
       ┌─────┴─────┐
       ▼           ▼
      MSE          R²
             │
             ▼
    Correlation Analysis
             │
             ▼
       Visualization
```

---

# 🎯 Problem Statement

House prices depend on multiple factors such as property size, number of bedrooms and bathrooms, construction year, location, condition, and garage availability.

The objective of this project is to build a Machine Learning model that learns the relationship between these property characteristics and their corresponding prices.

### Objective

> **Predict the price of a house using multiple property-related features.**

This is a **supervised regression problem** because the target variable, `Price`, is continuous.

---

# 📊 Dataset

The dataset contains **2,000 records and 10 columns**.

| Feature     | Type        | Description                       |
| ----------- | ----------- | --------------------------------- |
| `Id`        | Integer     | Unique property identifier        |
| `Area`      | Integer     | Property area                     |
| `Bedrooms`  | Integer     | Number of bedrooms                |
| `Bathrooms` | Integer     | Number of bathrooms               |
| `Floors`    | Integer     | Number of floors                  |
| `YearBuilt` | Integer     | Year the property was constructed |
| `Location`  | Categorical | Property location category        |
| `Condition` | Categorical | Property condition                |
| `Garage`    | Categorical | Garage category                   |
| `Price`     | Integer     | **Target variable**               |

### Dataset Characteristics

```text
Rows        : 2,000
Columns     : 10
Target      : Price
Problem     : Regression
Missing Data: None
```

---

# 🧠 Machine Learning Approach

## 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv("Datasets/house_price.csv")
```

Pandas provides the main data manipulation and analysis functionality used throughout the project.

---

## 2. Exploratory Data Analysis

The dataset is inspected using:

```python
df.head()
df.tail()
df.shape
df.describe()
df.info()
```

These operations help understand:

* Dataset structure
* Number of observations
* Feature types
* Statistical distributions
* Numerical ranges
* Potential data-quality issues

---

## 3. Data Quality Checks

The project checks for missing values:

```python
df.isnull().sum()
```

and duplicate records:

```python
df.duplicated().sum()
```

This provides an initial validation of dataset quality before model training.

---

# 🔄 Data Preprocessing

The dataset contains three categorical variables:

```text
Location
Condition
Garage
```

These variables cannot be directly processed by a standard Linear Regression implementation, so they are transformed into numerical representations using `LabelEncoder`.

Example:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Location"] = le.fit_transform(df["Location"])
df["Condition"] = le.fit_transform(df["Condition"])
df["Garage"] = le.fit_transform(df["Garage"])
```

After preprocessing, all model input features are numerical.

---

# 🎯 Feature and Target Selection

The prediction target is:

```python
y = df["Price"]
```

All remaining columns are used as input features:

```python
X = df.drop("Price", axis=1)
```

Conceptually:

```text
Input Features
─────────────────────────────
Area
Bedrooms
Bathrooms
Floors
YearBuilt
Location
Condition
Garage
Id
            │
            ▼
     Linear Regression
            │
            ▼
      Predicted Price
```

---

# ✂️ Train-Test Split

The dataset is divided into training and testing subsets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Configuration

| Parameter     | Value |
| ------------- | ----: |
| Training Data |   80% |
| Testing Data  |   20% |
| Random State  |    42 |

The training set is used to learn the relationship between features and price.

The testing set is used to evaluate how the trained model performs on unseen data.

---

# 🤖 Model

## Linear Regression

The project uses **Linear Regression** as its baseline regression model.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

The model estimates a linear relationship between the input variables and the target price.

Conceptually:

```text
X₁ ──┐
X₂ ──┤
X₃ ──┤
X₄ ──┤
X₅ ──┤──► Linear Regression ──► Predicted Price
X₆ ──┤
X₇ ──┤
X₈ ──┤
X₉ ──┘
```

---

# 🔮 Prediction

After training, predictions are generated using the test dataset:

```python
y_pred = model.predict(X_test)
```

The project compares the actual and predicted values:

```python
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
```

Example output structure:

| Actual Price |  Predicted Price |
| -----------: | ---------------: |
| Actual value | Model prediction |
| Actual value | Model prediction |
| Actual value | Model prediction |

---

# 📏 Model Evaluation

Two primary evaluation metrics are used.

## Mean Squared Error — MSE

```python
mse = mean_squared_error(y_test, y_pred)
```

MSE calculates the average squared difference between actual and predicted values.

**Lower MSE generally indicates better prediction performance.**

---

## R² Score

```python
r2 = r2_score(y_test, y_pred)
```

R² measures how much of the variation in the target variable is explained by the model.

A value closer to `1` generally indicates stronger explanatory performance.

### Important

The R² score should be interpreted together with the dataset, feature engineering, and other evaluation metrics rather than being treated as the only measure of model quality.

---

# 🔗 Correlation Analysis

The project also investigates relationships between numerical variables.

For example:

```python
df["Area"].corr(df["Price"])
```

and:

```python
df["Bedrooms"].corr(df["Price"])
```

A complete correlation matrix is generated using:

```python
df.corr(numeric_only=True)
```

### Pearson Correlation

```text
-1        0        +1
│─────────│─────────│
Negative  None    Positive
```

A correlation closer to:

* `+1` → strong positive linear relationship
* `0` → weak/no linear relationship
* `-1` → strong negative linear relationship

> **Correlation does not imply causation.**

---

# 📈 Visualization

The project visualizes the relationship between property area and price using a scatter plot:

```python
plt.scatter(df["Area"], df["Price"])

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs House Price")

plt.show()
```

This helps visually investigate whether larger properties tend to have higher prices.

---

# 🏗️ Project Architecture

```text
                 ┌─────────────────────┐
                 │     CSV Dataset     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Pandas DataFrame  │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Data Exploration &       │
              │ Quality Checks            │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Data Preprocessing       │
              │                          │
              │ Categorical Encoding     │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Feature / Target Split   │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Train / Test Split       │
              │        80 / 20           │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   Linear Regression      │
              │        Training          │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │       Prediction         │
              └────────────┬─────────────┘
                           │
                           ▼
             ┌────────────────────────────┐
             │       Evaluation           │
             │                            │
             │      MSE      │     R²     │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │ Correlation & Visualization│
             └────────────────────────────┘
```

---

# 📁 Repository Structure

```text
ML-1/
│
├── 📄 main.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
│
└── 📁 Datasets/
    └── 📄 house_price.csv
```

### File Description

| File               | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| `main.py`          | Main Machine Learning implementation            |
| `requirements.txt` | Python dependency versions                      |
| `README.md`        | Project documentation                           |
| `.gitignore`       | Prevents unnecessary files from being committed |
| `Datasets/`        | Contains the project dataset                    |

---

# ⚙️ Installation & Setup

## Prerequisites

Make sure Python 3 is installed.

Verify:

```bash
python --version
```

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Navigate into the project:

```bash
cd YOUR-REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run:

```bash
python main.py
```

The program will perform the complete workflow and display:

```text
Dataset information
        ↓
Data quality results
        ↓
Preprocessing information
        ↓
Train/Test dimensions
        ↓
Model training
        ↓
MSE
        ↓
R² Score
        ↓
Actual vs Predicted prices
        ↓
Correlation analysis
        ↓
Visualization
```

---

# 📦 Dependencies

The project uses:

```text
pandas
numpy
scikit-learn
matplotlib
```

Exact compatible versions are documented in `requirements.txt`.

---

# 🧪 Reproducibility

A fixed random seed is used:

```python
random_state=42
```

This ensures that the train-test split remains consistent across runs when the same environment and dataset are used.

---

# ⚠️ Current Limitations

This project is intentionally implemented as a **baseline Machine Learning model**. Several improvements can make it more robust for real-world use.

### 1. Label Encoding

Categorical variables are currently encoded using `LabelEncoder`.

For regression problems, **One-Hot Encoding** is often more appropriate for nominal categories because label encoding can introduce an artificial numerical ordering.

### 2. No Feature Scaling

The current implementation does not apply feature scaling.

Algorithms such as Linear Regression can sometimes benefit from appropriately scaled numerical features, particularly when comparing coefficients.

### 3. Baseline Model

Only Linear Regression is currently evaluated.

Other models could potentially capture non-linear relationships more effectively.

### 4. Limited Feature Engineering

The project currently uses the available features directly.

Potential derived features could include:

* Property age
* Area per bedroom
* Area per bathroom
* Age-condition interaction
* Location-based features

### 5. `Id` as a Feature

`Id` is an identifier rather than a meaningful property characteristic and should generally be evaluated carefully before being used as a predictive feature.

---

# 🚀 Future Improvements

The project can be extended into a more production-oriented ML pipeline.

### Data & Feature Engineering

* [ ] Remove non-predictive identifiers
* [ ] Use One-Hot Encoding
* [ ] Add feature engineering
* [ ] Handle outliers
* [ ] Apply feature scaling where appropriate

### Model Development

* [ ] Ridge Regression
* [ ] Lasso Regression
* [ ] Decision Tree Regression
* [ ] Random Forest Regression
* [ ] Gradient Boosting
* [ ] XGBoost
* [ ] Model comparison

### Evaluation

* [ ] MAE
* [ ] RMSE
* [ ] R²
* [ ] Cross-validation
* [ ] Residual analysis
* [ ] Hyperparameter tuning

### Deployment

* [ ] Save trained model using Joblib
* [ ] Build prediction API using FastAPI
* [ ] Create frontend interface
* [ ] Dockerize application
* [ ] Deploy model
* [ ] Add experiment tracking
* [ ] Add automated testing
* [ ] Create CI/CD pipeline

---

# 💡 Key Learning Outcomes

Through this project, the following Machine Learning concepts are demonstrated:

* Supervised Learning
* Regression
* Exploratory Data Analysis
* Data preprocessing
* Categorical encoding
* Feature-target separation
* Train-test splitting
* Linear Regression
* Model prediction
* MSE evaluation
* R² evaluation
* Correlation analysis
* Data visualization
* Basic ML project organization
* Reproducible experimentation

---

# 🔬 ML Workflow Summary

```text
                    MACHINE LEARNING PIPELINE

                           DATA
                            │
                            ▼
                    Data Exploration
                            │
                            ▼
                   Data Quality Checks
                            │
                            ▼
                    Preprocessing
                            │
                            ▼
                   Feature Engineering
                            │
                            ▼
                    Train / Test Split
                            │
                            ▼
                    Model Training
                            │
                            ▼
                      Prediction
                            │
                            ▼
                       Evaluation
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
                 MSE                  R²
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    Model Analysis
                            │
                            ▼
                    Future Improvement
```

---

# 📌 Conclusion

This project demonstrates the fundamental workflow required to develop a regression-based Machine Learning solution from a structured dataset.

Starting from data inspection and preprocessing, the project trains a Linear Regression model and evaluates its predictive performance using MSE and R². Correlation analysis and visualization are also used to understand relationships within the dataset.

The current implementation serves as a **baseline** that can be progressively improved through stronger preprocessing, feature engineering, model comparison, cross-validation, and deployment.

---

# 👨‍💻 Author

**Gopal**

Computer Science Engineering | Machine Learning & Software Development

---

## ⭐ Acknowledgement

This project was developed as part of Machine Learning practice and focuses on understanding the fundamentals of an end-to-end regression workflow.

---

<p align="center">

**If you found this project useful, consider giving it a ⭐ on GitHub.**

</p>
