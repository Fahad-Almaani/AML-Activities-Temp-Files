import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Salary_Data.csv')

#train test split
X = data[["YearsExperience"]]
Y = data[['Salary']]

# create the model
model = LinearRegression()
fit = model.fit(X, Y)

# predict the test set
pred_scores = fit.predict(X)

#Intercept (b0) and Coeffeicient (bn):
print("Intercept (b0):", model.intercept_)
print("Coefficient for age (b1):", model.coef_[0])

#SSE:
sse = ((Y - pred_scores) ** 2).sum()
print("Sum of Squered Errors: ", sse)

#R squared:
R = 1 - (sse/((Y - Y.mean()) ** 2).sum())
print("R squared:", R)

#Visualizing:
plt.scatter(X,Y, color="blue", label="Actual Data")
plt.plot(X, pred_scores, color="red", label= "Regression Line")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.legend()
plt.show
