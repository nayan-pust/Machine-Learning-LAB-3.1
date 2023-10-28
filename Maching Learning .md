# Machine-Learning-LAB-3.1

# ```Name : Naimur Rahman```
# ```Dept. of ICE(PUST)```

# programming Languages : ``` Python ```

# ```Project Name :``` Simple line Regression and Prediction for future Work


# First, we will import the three important libraries, which will help us for loading the dataset, plotting the graphs, and creating the Simple Linear Regression model.
```ruby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
# SetData Set or CSV file Import
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

# Calculate Mean Section and Size

```ruby 
mean_x = np.mean(x)
mean_y = np.mean(y)
#n = np.array(df)
#print(n)
siz = np.size(x)
print(siz)
```

# Line Regreetion Straight Line Equation thats Can you predict any Issues
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
# Prediction of test set result display the Output 

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


  



