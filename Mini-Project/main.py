import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Load the dataset
data = pd.read_csv("Video_Games_Sales.csv")

# Handle missing values (if any)
data.fillna(0, inplace=True)


target = 'Global_Sales'
# Encode categorical variables
data = pd.get_dummies(data, columns=['Platform', 'Genre', 'Publisher', 'Rating',"Developer"], drop_first=True)

# Split the data into training and testing sets
X = data.drop([target,"Name"], axis=1)
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
print(f"Random Forest vs XGBoost")
print(f"MSE: RF = {rf_mse}, XGB = {xgb_mse}")
print(f"R2: RF = {rf_r2}, XGB = {xgb_r2}")
print(f"MAE: RF = {rf_mae}, XGB = {xgb_mae}")
print(f"Explained Variance: RF = {rf_evs}, XGB = {xgb_evs}")