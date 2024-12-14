# Video Game Sales Analysis using Random Forest, XGBoost, and Ensemble Techniques

This project analyzes video game sales data using machine learning models: **Random Forest Regressor** and **XGBoost Regressor**, along with **Ensemble Techniques** (Averaging and Stacking). The objective is to predict the global sales of video games based on various features such as platform, genre, publisher, and other attributes.

---

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Code Walkthrough](#code-walkthrough)
   - [Library Imports](#library-imports)
   - [Loading and Cleaning Data](#loading-and-cleaning-data)
   - [Feature Encoding](#feature-encoding)
   - [Train-Test Split](#train-test-split)
   - [Random Forest Regressor](#random-forest-regressor)
   - [XGBoost Regressor](#xgboost-regressor)
   - [Averaging Ensemble](#averaging-ensemble)
   - [Stacking Ensemble](#stacking-ensemble)
   - [Model Evaluation](#model-evaluation)
   - [Feature Importances](#feature-importances)
   - [Actual vs Predicted Scatter Plot](#actual-vs-predicted-scatter-plot)
3. [How to Run the Code](#how-to-run-the-code)
4. [Results and Insights](#results-and-insights)

---

## Dataset Description
The dataset used in this project is `Video_Games_Sales.csv`. It contains information about video game sales, including:
- **Name**: Title of the game
- **Platform**: Gaming platform (e.g., PS4, X360, etc.)
- **Year_of_Release**: Release year
- **Genre**: Type of game (e.g., Action, Sports)
- **Publisher**: Company that published the game
- **Developer**: Game developer
- **Rating**: ESRB rating (e.g., E, M, T)
- **Global_Sales**: Total global sales in millions (target variable)

The goal is to predict `Global_Sales` using the other features.

---

## Code Walkthrough

### 1. Library Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
```
**Explanation**:
- **pandas**: For data manipulation and loading CSV files.
- **numpy**: For numerical operations.
- **matplotlib** and **seaborn**: For data visualization.
- **sklearn.model_selection**: To split the dataset into training and testing sets.
- **RandomForestRegressor**: Machine learning model for regression.
- **XGBRegressor**: XGBoost model for regression.
- **LinearRegression**: Used for stacking ensemble.
- **sklearn.metrics**: To evaluate model performance.

---

### 2. Loading and Cleaning Data
```python
# Load the dataset
data = pd.read_csv("Video_Games_Sales.csv")

# Handle missing values (if any)
data.fillna(0, inplace=True)
```
**Explanation**:
- The dataset is loaded using `pd.read_csv`.
- Missing values are replaced with `0` using `fillna` to avoid errors during model training.

---

### 3. Feature Encoding
```python
# Target variable
target = 'Global_Sales'

# Encode categorical variables
data = pd.get_dummies(data, columns=['Platform', 'Genre', 'Publisher', 'Rating', 'Developer'], drop_first=True)
```
**Explanation**:
- The `Global_Sales` column is the target variable.
- Categorical variables like `Platform`, `Genre`, `Publisher`, `Rating`, and `Developer` are converted into numerical features using **One-Hot Encoding** via `pd.get_dummies`. This ensures the models can interpret the data.

---

### 4. Train-Test Split
```python
# Split the data into training and testing sets
X = data.drop([target, "Name", "NA_Sales", "EU_Sales"], axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**Explanation**:
- Features (`X`) exclude `Global_Sales` (target) and unnecessary columns like `Name`, `NA_Sales`, and `EU_Sales`.
- Data is split into **training** (80%) and **testing** (20%) sets using `train_test_split`.

---

### 5. Random Forest Regressor
```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
```
**Explanation**:
- **Random Forest** is initialized with 100 trees.
- Model predictions are made on the test set.

---

### 6. XGBoost Regressor
```python
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
```
**Explanation**:
- **XGBoost** is initialized with 100 trees and `learning_rate=0.1`.

---

### 7. Averaging Ensemble
```python
ensemble_predictions_avg = (rf_predictions + xgb_predictions) / 2
```
**Explanation**:
- The predictions from Random Forest and XGBoost are averaged to improve model performance.

---

### 8. Stacking Ensemble
```python
stacked_train = np.column_stack((rf_model.predict(X_train), xgb_model.predict(X_train)))
stacked_test = np.column_stack((rf_predictions, xgb_predictions))
stacker = LinearRegression()
stacker.fit(stacked_train, y_train)
stacked_predictions = stacker.predict(stacked_test)
```
**Explanation**:
- Combines predictions from multiple models into a new dataset.
- **Linear Regression** is used to learn the relationships between model outputs and target values.

---

### 9. Model Evaluation
**Metrics for Comparison**:
- **MSE**: Mean Squared Error
- **R2 Score**: Model accuracy
- **MAE**: Mean Absolute Error
- **Explained Variance**

---

### 10. Feature Importances
**Random Forest and XGBoost Feature Importance Plots**:
- The top 10 features contributing to predictions are visualized using bar charts.

---

### 11. Actual vs Predicted Scatter Plot
```python
plt.scatter(y_test, rf_predictions, label='Random Forest')
plt.scatter(y_test, xgb_predictions, label='XGBoost')
plt.scatter(y_test, stacked_predictions, label='Stacking Ensemble')
```
**Explanation**:
- Visualizes how close predictions are to actual values.

---

## How to Run the Code
1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
2. Place `Video_Games_Sales.csv` in the same directory as the script.
3. Run the script:
   ```bash
   python script_name.py
   ```

---

## Results and Insights
1. **Ensemble Techniques** (Averaging and Stacking) improve predictive performance over individual models.
2. Feature importance analysis identifies key factors influencing video game sales.
3. Visualizations provide insights into model behavior and accuracy.

---

## Conclusion
This project demonstrates the use of regression models and ensemble techniques to predict video game sales. The structured approach highlights the benefits of combining models for better accuracy.
