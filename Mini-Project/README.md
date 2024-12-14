# Video Game Sales Analysis using Random Forest and XGBoost Regressors

This project analyzes video game sales data using two machine learning regression models: **Random Forest Regressor** and **XGBoost Regressor**. The objective is to predict the global sales of video games based on various features such as platform, genre, publisher, and other attributes.

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
   - [Model Evaluation](#model-evaluation)
   - [Feature Importances](#feature-importances)
   - [Actual vs Predicted Scatter Plot](#actual-vs-predicted-scatter-plot)
   - [Residual Plot](#residual-plot)
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
```

**Explanation**:

- **pandas**: For data manipulation and loading CSV files.
- **numpy**: For numerical operations.
- **matplotlib** and **seaborn**: For data visualization.
- **sklearn.model_selection**: To split the dataset into training and testing sets.
- **RandomForestRegressor**: Machine learning model for regression.
- **XGBRegressor**: XGBoost model for regression.
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
X = data.drop([target, "Name"], axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation**:

- Features (`X`) are all columns except `Global_Sales` (target) and `Name`.
- Data is split into **training** (80%) and **testing** (20%) sets using `train_test_split`.
- `random_state=42` ensures reproducibility.

---

### 5. Random Forest Regressor

```python
# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
```

**Explanation**:

- **Random Forest** is initialized with `n_estimators=100` (100 trees).
- The model is trained using the training data (`fit` method).
- Predictions are made on the test set.

---

### 6. XGBoost Regressor

```python
# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
```

**Explanation**:

- **XGBoost** is initialized with `n_estimators=100` and `learning_rate=0.1`.
- The model is trained on the training data and predictions are made on the test set.

---

### 7. Model Evaluation

```python
# Evaluate Random Forest
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_evs = explained_variance_score(y_test, rf_predictions)

# Evaluate XGBoost
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_evs = explained_variance_score(y_test, xgb_predictions)
```

**Metrics**:

- **MSE**: Mean Squared Error
- **R2 Score**: Model accuracy
- **MAE**: Mean Absolute Error
- **Explained Variance**: Variance explained by the model

---

### 8. Feature Importances

**Random Forest**:

```python
rf_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances[:10], y=X.columns[:10], palette='viridis')
```

**XGBoost**:

```python
xgb_importances = xgb_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importances[:10], y=X.columns[:10], palette='plasma')
```

**Explanation**:

- Feature importance shows which features contribute most to predictions.
- Top 10 important features are visualized for both models.

---

### 9. Actual vs Predicted Scatter Plot

```python
plt.scatter(y_test, rf_predictions, label='Random Forest')
plt.scatter(y_test, xgb_predictions, label='XGBoost')
```

**Explanation**:

- Compares actual values (`y_test`) against predictions for both models.
- A perfect prediction lies on the diagonal line.

---

### 10. Residual Plot

```python
xgb_residuals = y_test - xgb_predictions
sns.histplot(xgb_residuals, kde=True, color='orange')
```

**Explanation**:

- Residuals show the difference between actual and predicted values.
- A well-performing model will have residuals centered around zero.

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

1. Both **Random Forest** and **XGBoost** models are evaluated using MSE, R2 Score, MAE, and Explained Variance.
2. **Feature importance plots** highlight the most impactful features for predictions.
3. Visualization of predictions and residuals provides insights into model performance.

---

## Conclusion

This project demonstrates the use of two powerful regression models to predict video game sales and evaluates their performance using standard metrics and visualizations. The code is structured to be easily understandable and extendable for future analysis.
