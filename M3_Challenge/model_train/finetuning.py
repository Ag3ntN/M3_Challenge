import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

#read the CSV file
data = pd.read_csv('SEATTLE-EXPORT.csv').astype(float)

X = data.drop(columns=['Total_Homelessness'])  # DataFrame containing features
# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data['Total_Homelessness']  # Series containing target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
   tf.keras.layers.Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(9,)),
   tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
   tf.keras.layers.Dense(1)  # Output layer, no activation function for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=5, validation_split=0.2)

# Evaluate the model on test data
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test)
print(predictions)

#Evaluate model
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Save model weights as HDF5
model.save_weights("model_weights.h5")

#used to fine-tune the model, in the case if one model can be used for other city.
cityData = pd.read_csv('EXPORT SEATTLE.csv').astype(float)
city_model = tf.keras.models.clone_model(model)
X_city = cityData.drop(columns=['Total_Homelessness'])  # DataFrame containing features
y_city = cityData['Total_Homelessness']  # Series containing target variable
# Compile the city model
city_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

X_train_city, X_test_city, y_train_city, y_test_city = train_test_split(X_city, y_city, test_size=0.2, random_state=42)
# Train the city model on city-level data (fine-tuning)
city_model.fit(X_train_city, y_train_city, epochs=5, batch_size=32, validation_data=(X_test_city, y_test_city))
