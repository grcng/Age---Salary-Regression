# Age-vs-Salary-Regression
This Python script demonstrates how to fit a linear function to age vs. salary data using Gradient Descent. It's a simple yet powerful example of applying linear regression to analyze and model the relationship between two variables.

## Getting Started
To get started with this project, you'll need to have Python installed on your system. You'll also require the following Python packages:

`os`: Allows interaction with the operating system.  
`pandas`: Simplifies working with data uploaded from external sources.  
`sklearn`: Provides a linear regression package and tools for preprocessing.  
`matplotlib`: Used for creating plots and visualizations.

## Data Upload and Split
The script begins by uploading the dataset `data.csv`, and then it splits the data into training and testing sets. The feature (age) and target (salary) variables are also separated for both the training and testing sets.

## Data Normalization
Normalization of the feature (age) is performed to ensure that the data follows a standard scale, which is essential for Gradient Descent. The mean and standard deviation of the training data are used to normalize both the training and testing data.

## Linear Regression
The script uses the LinearRegression module from `sklearn` to fit a linear model to the training data. The model's performance is evaluated on both the training and testing data using standard error metrics.

## Polynomial Regression
In addition to linear regression, the script explores polynomial regression. It fits polynomial models of degrees 2 and 5 to the data, providing more flexibility in modeling the relationship between age and salary. The performance of these polynomial models is also assessed.

## Usage
You can run this Python script locally to see how linear and polynomial regression models can be used to analyze age vs. salary data. You can also modify the script to work with your own datasets or explore other machine learning techniques.

Enjoy exploring the relationships between variables and modeling data using this Python script!

