# Machine-Learning-LAB-3.1

# Simple Linear Regression in Machine Learning
Simple Linear Regression is a type of Regression algorithms that models the relationship between a dependent variable and a single independent variable. The relationship shown by a Simple Linear Regression model is linear or a sloped straight line, hence it is called Simple Linear Regression.

The key point in Simple Linear Regression is that the dependent variable must be a continuous/real value. However, the independent variable can be measured on continuous or categorical values.

Step-1: Data Pre-processing

The first step for creating the Simple Linear Regression model is data pre-processing.
We have already done it earlier in this tutorial. But there will be some changes, which are given in the below steps:

# Step 1 : First, we will import the three important libraries, which will help us for loading the dataset, plotting the graphs, and creating the Simple Linear Regression model.
```ruby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
# Step 2: SetData Set or CSV file Import
```ruby
# Read data from CSV file using pandas
#df = pd.read_csv('G:/Salary_data.csv')  # Replace 'your_data.csv' with your actual CSV file name

# Extract the features (x) and target (y) from the DataFrame
#x = np.array(df['A'])  # Replace 'Feature_Column' with the actual column name for features
#y = np.array(df['B'])   # Replace 'Target_Column' with the actual column name for target
x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

# Corresponding output data (target)
y = np.array([11, 13, 12, 15, 17, 18, 18, 19, 20, 22])
```
# Step 3 : After that, we need to extract the dependent and independent variables from the given dataset. The independent variable is years of experience, and the dependent variable is salary. Below is code for it:
```ruby

x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 1].values

```
# Step -5 :Calculate Mean Section and Size

```ruby 
mean_x = np.mean(x)
mean_y = np.mean(y)
#n = np.array(df)
#print(n)
siz = np.size(x)
print(siz)
```

# Step 5: Line Regreetion Straight Line Equation thats Can you predict any Issues
# Equations : # Calculate slope b and y-intercept a using the formula
# b = Σ(xi*yi)-n*(mean_x)(mean-y)/Σ(xi*x1)-n*(mean_x)^2
# a = ȳ - b * x̄
# Predict y values using the linear regression equation: y = a + bx

```ruby
numerator = np.sum(x*y) - siz*mean_x*mean_y
denominator = np.sum(x*x) - siz*mean_x*mean_x
slope = numerator / denominator
intercept = mean_y - slope * mean_x

print(numerator)
print(denominator)

predicted_y = slope * x + intercept
```
#  Prediction of test set result display the Output 

```ruby
# Plot the original data points
plt.scatter(x, y, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(x, predicted_y, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()

# Show the plot
plt.show()
```
# Final CODE

```ruby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from CSV file using pandas
#df = pd.read_csv('G:/Salary_data.csv')  # Replace 'your_data.csv' with your actual CSV file name

# Extract the features (x) and target (y) from the DataFrame
#x = np.array(df['A'])  # Replace 'Feature_Column' with the actual column name for features
#y = np.array(df['B'])   # Replace 'Target_Column' with the actual column name for target
x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

# Corresponding output data (target)
y = np.array([11, 13, 12, 15, 17, 18, 18, 19, 20, 22])
# Calculate the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)
#n = np.array(df)
#print(n)
siz = np.size(x)
print(siz)



# Calculate slope b and y-intercept a using the formula
# b = Σ(xi*yi)-n*(mean_x)(mean-y)/Σ(xi*x1)-n*(mean_x)^2
# a = ȳ - b * x̄
numerator = np.sum(x*y) - siz*mean_x*mean_y
denominator = np.sum(x*x) - siz*mean_x*mean_x
slope = numerator / denominator
intercept = mean_y - slope * mean_x
print(numerator)
print(denominator)


# Predict y values using the linear regression equation: y = mx + b
predicted_y = slope * x + intercept

# Plot the original data points
plt.scatter(x, y, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(x, predicted_y, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()

# Show the plot
plt.show()
```



# Graph on this Line Regreetion 
![image](https://github.com/nayan-pust/Machine-Learning-LAB-3.1/assets/114688354/29ec6c22-0349-46b7-90ab-a3f5170f14c7)


  



