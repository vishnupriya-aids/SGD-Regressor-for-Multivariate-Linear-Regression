# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and prepare the dataset with input features and target values (house price and number of       occupants).
2. Split the dataset into training and testing sets and apply feature scaling using StandardScaler.
3. Initialize and train the SGD Regressor models using the training data.
4. Predict house price and number of occupants for test and new inputs, evaluate performance using Mean Squared Error, and      display the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:VISHNUPRIYA E 
RegisterNumber:212225230308
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5]
])
y_price = np.array([30, 40, 45, 60, 75, 90])      
y_occupants = np.array([2, 3, 3, 4, 5, 6])
X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(
    X, y_price, y_occupants, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
price_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
occupant_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
price_model.fit(X_train_scaled, y_price_train)
occupant_model.fit(X_train_scaled, y_occ_train)
price_pred = price_model.predict(X_test_scaled)
occ_pred = occupant_model.predict(X_test_scaled)
print("House Price MSE:", mean_squared_error(y_price_test, price_pred))
print("Occupants MSE:", mean_squared_error(y_occ_test, occ_pred))
new_house = np.array([[1600, 4]])
new_house_scaled = scaler.transform(new_house)
predicted_price = price_model.predict(new_house_scaled)
predicted_occupants = occupant_model.predict(new_house_scaled)
print("\nPrediction for New House:")
print("Predicted House Price (in lakhs):", predicted_price[0])
print("Predicted Number of Occupants:", round(predicted_occupants[0])) 
*/
```

## Output:
<img width="1791" height="423" alt="Screenshot 2026-04-27 210150" src="https://github.com/user-attachments/assets/dcc52af4-c258-4860-a36c-8de7bbde2998" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
