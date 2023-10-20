# Python code to fit a linear function to the given data using Gradient Descent
# Import packages
import os  # enable interaction with the operating system
import pandas as pd  # simplifies working with data uploaded from external source
from sklearn.linear_model import LinearRegression  # linear regression package
from sklearn.preprocessing import PolynomialFeatures  # Polynomial
import matplotlib.pyplot as plt  # plots package

# Upload the data
DATA_FOLDER = '/Users/grcng/Downloads'
FILENAME = 'data.csv'
data = pd.read_csv(os.path.join(DATA_FOLDER, FILENAME))
print(data.head(n=10))  # print first 10 rows

# Split the data into train and test sets; also, split features (x) and targets (y)
dataTrain, dataTest = data.iloc[:10], data.iloc[10:20]
xRaw, yRaw = dataTrain.drop('Salary', axis=1), dataTrain.drop('Age', axis=1)
xRawTest, yRawTest = dataTest.drop('Salary', axis=1), dataTest.drop('Age', axis=1)

# Normalize the feature
xMean = xRaw.mean(axis=0)  # mean of training data
xStd = xRaw.std(axis=0, ddof=1)  # standard deviation of training data, ddof is for n-1 in the denominator
x = (xRaw - xMean) / xStd  # normalize the data
xTest = (xRawTest - xMean) / xStd

# Linear regression model
print('Linear Regression: ')

# Fit the model
model = LinearRegression()
model.fit(x, yRaw)  # find the best fit
print("Intercept:", model.intercept_[0])  # print the intercept
print("Slope coefficients:", model.coef_)  # print the coefficients

# Model performance
yFitted = model.predict(x)  # prediction for x values
errTrain = (yRaw - yFitted).std(ddof=1).values  # std of the error (get only value, not name of the column
print('Training data: ', errTrain)
yTestFitted = model.predict(xTest)
errTest = (yRawTest - yTestFitted).std(ddof=1).values
print('Test data: ', errTest)

# 2nd poly model
print('2nd Polynomial Regression: ')

# Fit the model
poly2 = PolynomialFeatures(degree=2)  # transform data into 2nd polynomial form
x2 = poly2.fit_transform(x)  # fit variable x
xTest2 = poly2.fit_transform(xTest)  # fit variable xTest
model2 = LinearRegression()
model2.fit(x2, yRaw)  # find the best fit

# Predict using the 2nd-degree polynomial regression model
yFitted2 = model2.predict(x2)

# Model performance
errTrain2 = (yRaw - yFitted2).std(ddof=1).values  # std of the error (get only value, not name of the column
print('Training data: ', errTrain2)
yTestFitted2 = model2.predict(xTest2)
errTest2 = (yRawTest - yTestFitted2).std(ddof=1).values
print('Test data: ', errTest2)

# 5th poly model
print('5th Polynomial Regression: ')

# Fit the model
poly5 = PolynomialFeatures(degree=5)  # transform data into 2nd polynomial form
x5 = poly5.fit_transform(x)  # fit variable x
xTest5 = poly5.fit_transform(xTest)  # fit variable xTest
model5 = LinearRegression()
model5.fit(x5, yRaw)  # find the best fit

# Predict using the 5th-degree polynomial regression model
yFitted5 = model5.predict(x5)

# Model performance
errTrain5 = (yRaw - yFitted5).std(ddof=1).values  # std of the error (get only value, not name of the column
print('Training data: ', errTrain5)
yTestFitted5 = model5.predict(xTest5)
errTest5 = (yRawTest - yTestFitted5).std(ddof=1).values
print('Test data: ', errTest5)

# Plot graph
plt.scatter(xRaw.values, yRaw.values, color='blue')  # original data
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')  # put the grid on

# Linear regression
plt.plot(xRaw, yFitted, color='red', linewidth=2, label='Linear Regression')  # fitted values

# Polynomial Regression
# Reshape and sort the values in xRaw
xRaw = xRaw.values.reshape(-1, 1)
xSorted = xRaw.argsort(axis=0)

# 2nd poly regression
plt.plot(xRaw[xSorted].squeeze(), yFitted2[xSorted].squeeze(), color='orange', linewidth=2,
         label='2nd Polynomial Regression')

# 5th poly regression
plt.plot(xRaw[xSorted].squeeze(), yFitted5[xSorted].squeeze(), color='green', linewidth=2,
         label='5th Polynomial Regression')

plt.title('Salary vs. age')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()  # Add a legend to the plot
plt.show()
