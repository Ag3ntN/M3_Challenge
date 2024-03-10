from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

data = pd.read_csv('EXPORT SEATTLE.csv')
data = data.astype(float)


X = data.drop(columns=['Total_Homelessness'])  # DataFrame containing features
y = data['Total_Homelessness']  # Series containing target variable
# Assuming X is the feature matrix (with multiple variables) and y is target variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the testing set
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)