# -*- coding: utf-8 -*-
"""Assignment1ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T-NOXS1nSJ2NpvuniIDUE2v0RMvq-8K6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""1. Linear Regression with nne Variables: (15 points)  
a. Implement linear regression to predict tumor_size (Tumor Size) using “mean_texture”
feature from dataset. “tumor_size” will be the target variable.  
b. Evaluate performance using metrics (such as Mean Squared Error (MSE), R-squared (R²),
and Adjusted R-squared (Adjusted R²). You may also use graphs for explaining your
observations.
"""

#read the data

data=pd.read_csv('Cancer_dataset.csv')
data.head(10)

#as per question 1 mean_texture and tumor size are selected
x_train=data[['mean_texture']].fillna(0)
y_train=data[['tumor_size']].fillna(0)

#Printing the variable and target
print(x_train)
print(y_train)

#convert the data into numpy array
X_train=np.array(x_train)
Y_train=np.array(y_train)

sh_x=x_train.shape[0]
index=np.arange(sh_x)
np.random.seed(42)
np.random.shuffle(index)
split=int(0.8*sh_x)
x_train=X_train[index[:split]]
y_train=Y_train[index[:split]]
x_test=X_train[index[split:]]
y_test=Y_train[index[split:]]

#Check both have same no of rows
print(f"x_train shape: {x_train.shape[0]}")
print(f"y_train shape: {y_train.shape[0]}")

plt.style.available

plt.style.use('seaborn-v0_8-dark')
plt.style.use('ggplot')

#To show how the values are scatted in the graph MEan texture vs tumor size
plt.scatter(x_train, y_train, marker='x', c='b')
plt.xlabel('Mean Texture')
plt.ylabel('Tumor Size')
plt.title('Mean Texture vs Tumor Size')
plt.show()

print(f"x_train: \n{x_train[0:10]}")
print(f"y_train: \n{y_train[0:10]}")
print(f"x_test: \n{x_test[0:10]}")
print(f"y_test: \n{y_test[0:10]}")

#now find w&b
"""use gradient descent and cost function to get the best w & b and calculate the prediction"""

w=0
b=0
learning_rate = 0.0001  # Try a smaller learning rate adjust it to get better result
tolerance = 10e-6  # Try a smaller tolerance value
m=x_train.shape[0] #to get the no of rows
previous_loss=float('inf') #infinity

for i in range(10000000): #we can change the iteration range here if the loop doesn't converge
  predicted_y=w*x_train+b #straight line for regression problems formula
  loss=(1/(2*m))*np.sum((predicted_y-y_train)**2) #costfunction

  if abs(previous_loss-loss)<tolerance: #if the loss is lesser than the tolerance it can be the global minima in the linear regression problem
    print(f"Converged at {i}th iteration")
    break

#updating the w and b till the above conditional statement satisfies
  dw=(1/m)*np.sum((predicted_y-y_train)*x_train)
  db=(1/m)*np.sum(predicted_y-y_train)

  w-=learning_rate*dw
  b-=learning_rate*db

#change the prevloss value by putting the found costfunction value in the previous loss
  previous_loss=loss

print(f"w={w}")
print(f"b={b}")

final_value=w*x_train+b #linear regression st line formula
print(f"Final value: \n{final_value[:10]}")

#plot the straight line found in the graph to see how well it fits the data
plt.scatter(x_train, y_train, marker='x',c='b', label='Actual Data')
plt.plot(x_train, final_value, c='r', label='Regression line( best fit)')
plt.xlabel('Mean Texture')
plt.ylabel('Tumor Size')
plt.title('Mean Texture vs Tumor Size')
plt.legend()
plt.show()

final_value_x=w*x_test+b
print(f"Final value: {final_value_x[0:10]}")
# Calculate Mean Squared Error (MSE) Loss
MSE_loss = (1 / (2 * m)) * np.sum((final_value_x - y_test) ** 2)
print(f"Mean Squared Error (MSE) Loss: {round(MSE_loss,4)}")
# Calculate Total Sum of Squares (TSS)
y_mean = np.mean(y_train)
print(f"MEan: {y_mean}")
print(f"y_train: {y_train[:5]}")
TSS = np.sum((y_test - y_mean) ** 2)

# Calculate Residual Sum of Squares (RSS)
RSS = np.sum((y_test - final_value_x) ** 2)

# Calculate R-squared (R²)
r_squared = 1 - (RSS / TSS)

print(f"R-squared: {round(r_squared,4)}")

#calculate adjusted R-Squared
n=x_train.shape[0]
p=1#there is only one predictor for linear regression
adjus_rSqu=1-(((1-r_squared)*(n-1))/(n-p-1))
print(f"Adjusted R-Squared: {round(adjus_rSqu,4)}")

predic=w*x_test[0]+b
print(f"Predicted value: {predic}")

#checking the pattern
residual=y_train-final_value
plt.scatter(y_train, residual, marker='x', c='b')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual target values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()