import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#read the CSV file
data = pd.read_csv('albq final final.csv').astype(float)

X = data.drop(columns=['Total_Homelessness'])  # DataFrame containing features
# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data['Total_Homelessness']  # Series containing target variable
y = y.to_numpy()
target_scaler = MinMaxScaler()
y_normalized = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(9,)),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='relu')  # Output layer, no activation function for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=70, batch_size=2, validation_split=0.2)


# Evaluate the model on test data
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
prediction_normalized = model.predict(X_test)
prediction = target_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()
print("FIRST: ")
print(prediction)

#Evaluate model
mae = mean_absolute_error(y_test, prediction)
print("Mean Absolute Error:", mae)

# Save model architecture as JSON
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights as HDF5
model.save("final_albq.h5")

weights = model.get_weights()
feature_importance = np.sum(np.abs(weights[0]), axis=1)
print(feature_importance)



#read the CSV file
data = pd.read_csv('TESTGENALBQ.csv').astype(float)

X = data.drop(columns=['Total_Homelessness'])  # DataFrame containing features
# Standardize the data
X = scaler.fit_transform(X)
y = data['Total_Homelessness']  # Series containing target variable
y = y.to_numpy()
y_normalized = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)


# Compile the model

print("Test Loss:", loss)
print(X_test)
# Make predictions
prediction_normalized = model.predict(X_test)
prediction = target_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()
print(prediction)