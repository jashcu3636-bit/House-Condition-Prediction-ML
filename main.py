import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Dataset/house_price.csv")


print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== DESCRIPTION ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

print("\n========== ORIGINAL INFO ==========")
df.info()


# ==========================================
# 2. ENCODE CATEGORICAL DATA
# ==========================================

le = LabelEncoder()

df["Location"] = le.fit_transform(df["Location"])
df["Condition"] = le.fit_transform(df["Condition"])
df["Garage"] = le.fit_transform(df["Garage"])


print("\n========== AFTER ENCODING ==========")
df.info()


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Price", axis=1)
y = df["Price"]


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 5. CREATE LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. MEAN SQUARED ERROR
# ==========================================

mse = mean_squared_error(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Mean Squared Error:", mse)


# ==========================================
# 9. R2 SCORE
# ==========================================

r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)


# ==========================================
# 10. ACTUAL VS PREDICTED
# ==========================================

result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(result.head(10))


# ==========================================
# 11. CORRELATION
# ==========================================

print("\n========== CORRELATION ==========")

print("Area vs Price:",
      df["Area"].corr(df["Price"]))

print("Bedrooms vs Price:",
      df["Bedrooms"].corr(df["Price"]))

print("\nComplete Correlation Matrix:")
print(df.corr(numeric_only=True))


# ==========================================
# 12. SCATTER PLOT
# ==========================================

plt.scatter(df["Area"], df["Price"])

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs House Price")

plt.show()