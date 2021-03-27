# Import the dependencies
import pandas as pd 
from matplotlib import pyplot as plt 
import numpy as np
import statsmodels.api as sm

# Import the crime dataset
file_path = './data/crime_data.xlsx'
crime_df = pd.read_excel(file_path)
crime_df.head()

# Import the donor2008 dataset
file = './data/donors2008.csv'
donor_df = pd.read_csv(file, encoding = "ISO-8859-1")
donor_df

# Cleaning the dataset
del donor_df['FIELD8']
donor_df = donor_df.dropna(how='any')
donor_df['Amount'] = pd.to_numeric(donor_df['Amount'])
donor_df['Employer'] = donor_df['Employer'].replace(
    {'Self Employed': 'Self-Employed', 'Self': 'Self-Employed'})
donor_df.describe()

# Create data visualization
# Histogram for donor amount
plt.hist(donor_df['Amount'])
plt.title("Histogram for Donor Amount")
plt.xlabel("Donor Amount ($)")
plt.ylabel("Frequency")
plt.show()

# Scatterplot for rape and robbery
plt.scatter(crime_df['rape'], crime_df['robbery'])
plt.title("Scatter Plot: Rape vs. Robbery (Annual Data)")
plt.xlabel("Reported Rape Cases")
plt.ylabel("Reported Robbery Cases")
plt.show()

# Simple Linear Regression Analysis

# Import the advertisng dataset
file_path = './data/advertising.csv'
adv_df = pd.read_csv(file_path, sep=',')
adv_df = adv_df.drop(adv_df.columns[0], axis=1)
adv_df.head()

# Check the corelations between each variables
adv_df.corr()

# Build the regression model / OLS model based on TV advertising budget

# Step 1: Create the dependent and independent variables
X = adv_df['TV']
X = sm.add_constant(X)
y = adv_df["sales"]

# Step 2: Fitting the data to the regression model
fit = sm.OLS(y, X).fit()

# Step 3: Print the OLS regresion report
print(fit.summary())

# Extracting the parameters and statistics from the report
print('Parameters: ', fit.params)
print('Standard Errors: ', fit.bse)
print('R^2: ', fit.rsquared)

# Making in-sample prediction based on the model fit
print('Predicted Values: ', fit.predict())

# Plot the regression model with data
from statsmodels.sandbox.regression.predstd import wls_prediction_std

pred, iv_l, iv_u = wls_prediction_std(fit)
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(adv_df["TV"], y, 'o', label = "data")
ax.plot(adv_df["TV"], fit.fittedvalues, 'r--', label = "OLS")
ax.plot(adv_df["TV"], iv_u, 'r--')
ax.plot(adv_df["TV"], iv_l, 'r--')
ax.legend(loc = 'best');