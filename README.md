# 🏠 House Price Prediction ML

A simple Machine Learning project that predicts house prices using **Linear Regression**.

## 📌 Project Overview

The project takes house-related features such as area, bedrooms, bathrooms, floors, year built, location, condition, and garage information and predicts the house **Price**.

## 🧠 Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Categorical Encoding
   ↓
Train-Test Split (80/20)
   ↓
Linear Regression
   ↓
Prediction
   ↓
MSE + R² Score
   ↓
Correlation Analysis & Visualization
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 🤖 Machine Learning Model

**Linear Regression** is used for this regression problem.

The model is evaluated using:

- Mean Squared Error (MSE)
- R² Score

## 📂 Project Structure

```text
House-Price-Prediction-ML/
│
├── Dataset/
│   └── house_price.csv
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

The program displays dataset information, model evaluation results, correlation analysis, and a house area vs. price visualization.

## 🎯 Learning Outcomes

- Data preprocessing
- Categorical encoding
- Train-test splitting
- Linear Regression
- Regression evaluation
- Correlation analysis
- Data visualization

## 👨‍💻 Author

**Bala Ji**

GitHub: https://github.com/balaji0307-om
