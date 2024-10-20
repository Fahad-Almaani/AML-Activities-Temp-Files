# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Visualization library
import matplotlib.pyplot as plt

# Load the Digits dataset
digits_data = load_digits()
x = digits_data.data   # Features (image pixel data)
y = digits_data.target # Labels (digits 0 to 9)


# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # k = 5 (number of neighbors)

# Train the classifier
knn.fit(x_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize some of the test data and predictions
plt.figure(figsize=(10, 4))
for i in range(1, 8):
    plt.subplot(1, 7, i)
    plt.imshow(x_test[i].reshape(8, 8), cmap='gray')  # Digits images are 8x8 pixels
    plt.title(f'Label: {y_test[i]}\nPred: {y_pred[i]}')
    plt.axis('off')
plt.suptitle(f"Accuracy: {accuracy_score(y_test, y_pred)}")
plt.show()

print(len(x[1]
),y[1])