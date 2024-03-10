import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)  # One feature
y = 2 * X.squeeze() + np.random.normal(0, 0.1, size=100)  # Linear relationship with noise

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Print R-squared
print("R-squared:", r2)