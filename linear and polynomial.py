#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# 
# 

# In[3]:


# Dummy data
import numpy as np
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Number of hours
marks = np.array([20, 22, 37, 41, 51, 58, 65, 68, 80, 89])  # Marks obtained

print(hours)
print(marks)


# In[4]:


# Plot the actual data
import matplotlib.pyplot as plt
plt.plot(hours, marks, color='red')
plt.xlabel('Number of Hours')
plt.ylabel('Marks Obtained')
plt.title('Shape of the Data')
plt.show()


# hours = [ 1  2  3  4  5  6  7  8  9 10]

# In[5]:


# Reshape the input data to 2D
#axis 1
hours = hours.reshape(-1, 1)
print(hours)


# In[6]:


# Creating linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# Training Model
model.fit(hours, marks)


# In[7]:


# Calculate predictions
predicted_marks = model.predict(hours)
print("Hours", hours)
print("Actual Marks", marks)
print("Predicted Marks", predicted_marks)


# In[8]:


# Plot the model line with actual data
plt.scatter(hours, marks, color='blue', label='Actual Data Points')
#plt.plot(hours, marks, color='green', label='Actual Line')
plt.plot(hours, predicted_marks, color='red', label='Linear Regression Model')
plt.xlabel('Number of Hours')
plt.ylabel('Marks Obtained')
plt.title('Linear Regression Model')
plt.legend()
plt.show()


# In[9]:


# Calculate MSE and RMSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(marks, predicted_marks) #actual - predicted 
rmse = np.sqrt(mse)
# Print MSE and RMSE
print('MSE:', mse)
print('RMSE:', rmse)


# # Polynomial Regression or Least Square 

# In[10]:


from sklearn.preprocessing import PolynomialFeatures


# In[19]:


hours_poly = PolynomialFeatures(degree=9).fit_transform(hours)
hours_poly


# In[20]:


poly_model = LinearRegression()
poly_model.fit(hours_poly, marks)


# In[21]:


# Calculate predictions
predicted_marks = poly_model.predict(hours_poly)
print("Hours", hours)
print("Actual Marks", marks)
print("Predicted Marks", predicted_marks)


# In[22]:


# Plot the model line with actual data
plt.scatter(hours, marks, color='blue', label='Actual Data')
#plt.plot(hours, marks, color='green', label='Actual Line')
plt.plot(hours, predicted_marks, color='red', label='Linear Regression Model')
plt.xlabel('Number of Hours')
plt.ylabel('Marks Obtained')
plt.title('Ploynomial Regression Model')
plt.legend()
plt.show()

