import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('salary_data2.csv')

#train test split
data = data.dropna()
X = data.drop(columns=['SALARY',"FIRST NAME","LAST NAME","CURRENT DATE","DOJ"])
Y = data[['SALARY']]

# drop missing values
# encode the UNIT column
X = pd.get_dummies(X, columns=['UNIT',"DESIGNATION","SEX"], drop_first=True)

# create the model
model = LinearRegression()
fit = model.fit(X, Y)

# predict the test set
pred_scores = fit.predict(X)

#Intercept (b0) and Coeffeicient (bn):
print("Intercept (b0):", model.intercept_)
print("Coefficient for age (b1):", model.coef_[0])

#MSE:
mse = mean_squared_error(Y, pred_scores)
print("Mean Squered Errors: ", mse)

#R squared:
R = r2_score(Y,pred_scores)
print("R squared:", R)

#MAE:
mae = mean_absolute_error(Y, pred_scores)
print("Mean Absolute Errors:", mae)

plt.figure(figsize=(10, 6))
plt.scatter(Y, pred_scores, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--', label='Perfect Fit')  # Line of perfect prediction
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.grid(True)
plt.show()
