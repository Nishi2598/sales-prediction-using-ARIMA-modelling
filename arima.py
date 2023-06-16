# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:18:29 2023

@author: nishi
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization
import seaborn as sns     

path = 'C:/Users/Owner/anaconda3/dataset.txt'

df = pd.read_csv(path)

df.head()

print(df)

from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.value.dropna()) # dropping NaN values in 'value'
print('ADF Statistic: %f' % result[0]) # ADF statistic value more lower value, more stronger evidence of presents of unit root and suggests stationarity
print('p-value: %f' % result[1]) #p-value less than 0.05, strong evidence of stationarity 

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True) #plot 3x2 size with all element sharing same axis
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series') # plotting original time series
plot_acf(df.value, ax=axes[0, 1]) # plotting auto-correlation function

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing') # plotting 1st order differencing using .diff()
plot_acf(df.value.diff().dropna(), ax=axes[1, 1]) #plotting ACF plot and dropping missing values if any

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing') #plotting 2nd order differencing
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1]) #plotting ACF plot

plt.show()

#We can see that the time series reaches stationarity with two orders of differencing

# We will find the AR value by PACF plot. In a stationary series, the autocorrelation should ideally be close to zero at all lags. 
# However, if autocorrelation is present in a stationarized series, it suggests that there is still some dependency between current and past observations that the model hasn't captured.

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1]) # The PACF measures the correlation between the current observation and its lagged values, while controlling for the intermediate lags. 

plt.show()

# We can see that the PACF lag 1 is quite significant since it is well above the significance line. So, we will fix the value of p as 1.

# ACF plot is for the number of MA terms
# Plotting ACF plot for 1st differenced series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# We can see that couple of lags are well above the significance line. So, we will fix q as 2.

# Now we have p, q and d values, we will build ARIMA model.
from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model with values of 1st differenced series
model = ARIMA(df.value, order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# The table in the middle is the coefficients table where the values under ‘coef’ are the weights of the respective terms.

# The coefficient of the MA2 term is close to zero and the P-Value in ‘P>|z|’ column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.

# So, we will rebuild the model without the MA2 term.
# 1,1,1 ARIMA Model
model = ARIMA(df.value, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# The model AIC has slightly reduced, which is good. The p-values of the AR1 and MA1 terms have improved and are highly significant (<< 0.05).

# Let’s plot the residuals to ensure there are no patterns (that is, look for constant mean and variance).
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1]) #This plots the density of the residuals as a KDE plot on the second subplot (ax[1])
plt.show()

# A well-fitted model would have residuals that appear random with no obvious patterns, and the density plot would resemble a symmetric bell-shaped curve.

# Actual vs Fitted
model_fit.plot_predict(dynamic=False) #the model gets trained up until the previous value to make the next prediction.
plt.show()

# Finding optimal ARIMA model using Out-of-time cross validation. That is forcasting future values.
# In Out-of-Time cross-validation, we move backwards in time and forecast into the future to as many steps we took back. Then we compare the forecast against the actuals.
from statsmodels.tsa.stattools import acf

# Create Training and Test
train = df.value[:85]
test = df.value[85:]

# Build Model
model = ARIMA(train, order=(1,1,1)) 
fitted = model.fit(disp=-1) 
fc, se, conf = fitted.forecast(119, alpha=0.05)  # forecasting 119 values with 95% confidence level    
# Make as pandas series
fc_series = pd.Series(fc, index=test.index) # This creates a pandas Series object (fc_series) to store the forecasted values (fc), with the index set to match the index of the test data.
lower_series = pd.Series(conf[:, 0], index=test.index) # This creates a pandas Series object (lower_series) to store the lower bound of the confidence intervals (conf[:, 0]), with the index matching the test data index.
upper_series = pd.Series(conf[:, 1], index=test.index) # This creates a pandas Series object (upper_series) to store the upper bound of the confidence intervals (conf[:, 1]), with the index matching the test data index.

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# We can see that the predicted forecasts is consistently below the actuals. That means, by adding a small constant to our forecast, the accuracy will certainly improve.
# Build Model
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(119, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('months')
plt.ylabel('sales in thousands')
plt.show()

# The AIC has reduced to 245 from 843 which is good. Mostly, the p-values of the X terms are less than < 0.05, which is great. So overall this model is much better.
# printing forecasted values
print(fc)

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

print(forecast_accuracy(fc, test.values))

#Around 23.22% MAPE implies the model is about 76.78% accurate in predicting the next 119 observations.

 
