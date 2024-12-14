import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Load the dataset
data = pd.read_csv("Video_Games_Sales.csv")

# Handle missing values (if any)
data.fillna(0, inplace=True)

# Target variable
target = 'Global_Sales'

# Encode categorical variables
data = pd.get_dummies(data, columns=['Platform', 'Genre', 'Publisher', 'Rating', 'Developer'], drop_first=True)

# # Visualize the correlation heatmap before splitting data
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
# plt.title('Feature Correlation Heatmap')
# plt.show()    

# Split the data into training and testing sets
X = data.drop([target, "Name","NA_Sales","EU_Sales"], axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Averaging Ensemble
ensemble_predictions_avg = (rf_predictions + xgb_predictions) / 2

# Evaluate Averaging Ensemble
ensemble_mse_avg = mean_squared_error(y_test, ensemble_predictions_avg)
ensemble_r2_avg = r2_score(y_test, ensemble_predictions_avg)
ensemble_mae_avg = mean_absolute_error(y_test, ensemble_predictions_avg)
ensemble_evs_avg = explained_variance_score(y_test, ensemble_predictions_avg)

print("\nAveraging Ensemble Results:")
print(f"MSE: {ensemble_mse_avg}")
print(f"R2 Score: {ensemble_r2_avg}")
print(f"MAE: {ensemble_mae_avg}")
print(f"Explained Variance Score: {ensemble_evs_avg}")

# Stacking Ensemble
stacked_train = np.column_stack((rf_model.predict(X_train), xgb_model.predict(X_train)))
stacked_test = np.column_stack((rf_predictions, xgb_predictions))

stacker = LinearRegression()
stacker.fit(stacked_train, y_train)
stacked_predictions = stacker.predict(stacked_test)

# Evaluate Stacking Ensemble
stacked_mse = mean_squared_error(y_test, stacked_predictions)
stacked_r2 = r2_score(y_test, stacked_predictions)
stacked_mae = mean_absolute_error(y_test, stacked_predictions)
stacked_evs = explained_variance_score(y_test, stacked_predictions)

print("\nStacking Ensemble Results:")
print(f"MSE: {stacked_mse}")
print(f"R2 Score: {stacked_r2}")
print(f"MAE: {stacked_mae}")
print(f"Explained Variance Score: {stacked_evs}")

# Model Comparison
print("\nModel Comparison:")
print(f"Random Forest R2: {r2_score(y_test, rf_predictions)}")
print(f"XGBoost R2: {r2_score(y_test, xgb_predictions)}")
print(f"Averaging Ensemble R2: {ensemble_r2_avg}")
print(f"Stacking Ensemble R2: {stacked_r2}")

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
plt.scatter(y_test, stacked_predictions, color='green', alpha=0.5, label='Stacking Ensemble')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual vs Predicted Global Sales')
plt.legend()
plt.show()
