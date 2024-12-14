import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Load the dataset
data = pd.read_csv("Video_Games_Sales.csv")

# Handle missing values (if any)
data.fillna(0, inplace=True)

# Target variable
target = 'Global_Sales'

# Encode categorical variables
data = pd.get_dummies(data, columns=['Platform', 'Genre', 'Publisher', 'Rating', 'Developer'], drop_first=True)



# Split the data into training and testing sets
X = data.drop([target, "Name"], axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_evs = explained_variance_score(y_test, rf_predictions)
print("Random Forest Results:")
print(f"MSE: {rf_mse}")
print(f"R2 Score: {rf_r2}")
print(f"MAE: {rf_mae}")
print(f"Explained Variance Score: {rf_evs}")

# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Evaluate XGBoost
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_evs = explained_variance_score(y_test, xgb_predictions)
print("\nXGBoost Results:")
print(f"MSE: {xgb_mse}")
print(f"R2 Score: {xgb_r2}")
print(f"MAE: {xgb_mae}")
print(f"Explained Variance Score: {xgb_evs}")

# Comparison
print("\nModel Comparison:")
print(f"MSE: RF = {rf_mse}, XGB = {xgb_mse}")
print(f"R2: RF = {rf_r2}, XGB = {xgb_r2}")
print(f"MAE: RF = {rf_mae}, XGB = {xgb_mae}")
print(f"Explained Variance: RF = {rf_evs}, XGB = {xgb_evs}")

# Feature Importances - Random Forest
rf_importances = rf_model.feature_importances_
indices_rf = np.argsort(rf_importances)[::-1]
features_rf = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances[indices_rf][:10], y=features_rf[indices_rf][:10], palette='viridis')
plt.title('Top 10 Important Features - Random Forest')
plt.show()

# Feature Importances - XGBoost
xgb_importances = xgb_model.feature_importances_
indices_xgb = np.argsort(xgb_importances)[::-1]
features_xgb = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importances[indices_xgb][:10], y=features_xgb[indices_xgb][:10], palette='plasma')
plt.title('Top 10 Important Features - XGBoost')
plt.show()

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_predictions, color='blue', alpha=0.5, label='Random Forest')
plt.scatter(y_test, xgb_predictions, color='red', alpha=0.5, label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual vs Predicted Global Sales')
plt.legend()
plt.show()

# Residual Plot for XGBoost
xgb_residuals = y_test - xgb_predictions
plt.figure(figsize=(8, 6))
sns.histplot(xgb_residuals, kde=True, color='orange')
plt.title('Residuals Distribution - XGBoost')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()