import pandas as pd
import sklearn.model_selection as skms
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# Load the data
data = pd.read_csv('car_data.csv')


# remove the User ID column
data = data.drop('User ID', axis=1)
# clean na values
data = data.dropna()
data = pd.get_dummies(data, columns=['Gender'])
#train test split
X = data.drop('Purchased', axis=1)
Y = data['Purchased']
X_train, X_test, Y_train, Y_test = skms.train_test_split(X, Y, test_size=0.25)

# create the model

dt = DecisionTreeClassifier()
fit = dt.fit(X_train, Y_train)

# predict the test set
Y_pred = fit.predict(X_test)

# print the accuracy
print(accuracy_score(Y_test, Y_pred))

# create a confusion matrix
confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()






