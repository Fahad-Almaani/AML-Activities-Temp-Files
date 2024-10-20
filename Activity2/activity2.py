import pandas as pd
import sklearn.model_selection as skms
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
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

dt = DecisionTreeClassifier(max_depth=3)
fit = dt.fit(X_train, Y_train)

# predict the test set
Y_pred = fit.predict(X_test)

# print the accuracy
print(f"{accuracy_score(Y_test, Y_pred)*100}%")

# visualize the decision tree

plt.figure(figsize=(12,12))
tree.plot_tree(dt, feature_names=X.columns, class_names=['0', '1'], filled=True)
plt.show()


# confusion matrix 
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True)
plt.show()


