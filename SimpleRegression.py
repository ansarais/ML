import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('house.csv')



# Selection of Regression model
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()

# Fitting the model

regression_model.fit(dataset[['Area']], dataset['Price'])

# Predicting the result
predicted_price = regression_model.predict([[2500]])
print(predicted_price)
# Visualising the Training set results
plt.scatter(dataset['Area'], dataset['Price'], color='red') 
plt.plot(dataset['Area'], regression_model.predict(dataset[['Area']]), color='blue')
plt.title('Area vs Price (Training set)')   
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
