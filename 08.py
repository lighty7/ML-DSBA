import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target
# Convert the data to a pandas DataFrame for easier manipulation
boston_df = pd.DataFrame(data=X, columns=boston.feature_names)
boston_df['target'] = y
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize Linear Regression model
linear_regression = LinearRegression()
# Train the model
linear_regression.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = linear_regression.predict(X_test)
# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)