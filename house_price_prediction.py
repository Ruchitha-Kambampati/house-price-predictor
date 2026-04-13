import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score


# =========================
# LOAD DATASET
# =========================
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

print("Dataset Loaded Successfully")
print(df.head())


# =========================
# SPLIT DATA
# =========================
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# MODEL TRAINING
# =========================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# =========================
# HYPERPARAMETER TUNING
# =========================
params = {
    'n_estimators': [100, 200],
    'max_depth': [10, None]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("\nBest Parameters:", grid.best_params_)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.4f}")


print("\nModel Performance:")
evaluate(y_test, lr_pred, "Linear Regression")
evaluate(y_test, dt_pred, "Decision Tree")
evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, best_pred, "Tuned Random Forest")

# =========================
# SAMPLE PREDICTION
# =========================
sample = X_test.iloc[0:1]
prediction = best_model.predict(sample)

print("\nSample Prediction:")
print("Predicted Price:", prediction[0])
print("Actual Price:", y_test.iloc[0])
# =========================
# USER INPUT PREDICTION
# =========================

print("\nEnter values for prediction:")

user_data = []

for col in X.columns:
    value = float(input(f"Enter {col}: "))
    user_data.append(value)

# Convert to DataFrame
user_df = pd.DataFrame([user_data], columns=X.columns)

# Predict
prediction = best_model.predict(user_df)

print("\nPredicted House Price(in $1,00,000):", prediction[0])



# =========================
# FEATURE IMPORTANCE
# =========================
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
