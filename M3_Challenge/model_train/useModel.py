import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pandas as pd

# Load the trained model
loaded_model = tf.keras.models.load_model('final_albq.h5')
data = pd.read_csv('TESTGENALBQ.csv').astype(float)
new_data = data.drop(columns=['Total_Homelessness']) 

# Assuming you have new data stored in a variable called new_data
# Standardize the new data using the same scaler used for training
scaler = StandardScaler()  # Use the same scaler you used for training
scaler = scaler.fit(pd.read_csv('TESTGENALBQ.csv').astype(float))

new_data_scaled = scaler.transform(new_data)

# Make predictions
prediction_normalized = loaded_model.predict(new_data_scaled)
# Inverse transform the normalized predictions to get the actual values
target_scaler = MinMaxScaler()  # Use the same scaler you used for training the target variable
y = new_data['Total_Homelessness']  # Series containing target variable
y = y.to_numpy()
target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
prediction = target_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()

print(prediction)
