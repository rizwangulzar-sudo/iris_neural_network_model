#Steps:
# Prepare the Data
# Build the Neural Network Model
# Train the Model
# Evaluate the Model

#Install Dependencies
#pip install tensorflow

#02 Prepare the Data
#For simplicity, we will use the Iris dataset from sklearn. 
# This dataset contains data about flowers with 4 features (like petal length, petal width, etc.) and 3 target classes. 
# For simplicity, we'll only work with 2 classes: "Setosa" and "Versicolor" (binary classification).

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_iris()
X = data.data
y = (data.target != 0) * 1  # 1 if Versicolor or Virginica, 0 if Setosa

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build the Neural Network Model

# 03 Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation='relu'),  # Input layer and first hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')            # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Input Layer: 4 features (from the Iris dataset)
# Hidden Layer: 8 neurons, ReLU activation function
# Output Layer: 1 neuron, Sigmoid activation (for binary classification)

# 03 Train the Model
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# 04 Evaluate the Model
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 05 Make Predictions
# Predict on new data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

print("Predictions:", y_pred[:10])  # Display the first 10 predictions

# Summary of Neural Network Steps:
# Input Layer: 4 features from the dataset.
# Hidden Layer: 8 neurons, ReLU activation.
# Output Layer: 1 neuron, Sigmoid activation (gives the probability of class 1).
# Training: We used binary_crossentropy as the loss function and accuracy as the metric.

# 06 Visualization of Training
#To visualize how well the model is performing, you can plot the loss and accuracy curves:
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Key Points:
# Activation Functions: ReLU in the hidden layer and Sigmoid in the output layer.
# Loss Function: Binary Cross-Entropy for binary classification.
# Optimizer: Adam optimizer (works well for most cases).








