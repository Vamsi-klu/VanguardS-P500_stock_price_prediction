#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install pmdarima


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os


# In[6]:


from pmdarima.arima import auto_arima


# In[7]:


from IPython.core.debugger import set_trace

df = pd.read_csv('VOO_HistoricalData_1626316081093.csv')


# In[8]:


df.head()


# In[9]:


df = df[['Close']].copy()


# In[10]:


df.head


# In[11]:


#checking wether if price seris is stationary or not
from statsmodels.tsa.stattools import adfuller

result = adfuller(df.Close.dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


# In[12]:


from statsmodels.graphics.tsaplots import plot_acf


# In[15]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(df.Close)
ax1.set_title("Original")
# add ; at the end of the plot function so that the plot is not duplicated
plot_acf(df.Close, ax=ax2)


# In[17]:


diff = df.Close.diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(diff)
ax1.set_title("Difference once")
plot_acf(diff, ax=ax2)


# In[18]:


diff = df.Close.diff().diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(diff)
ax1.set_title("Difference twice")
plot_acf(diff, ax=ax2)


# In[19]:


from pmdarima.arima.utils import ndiffs


# In[21]:


ndiffs(df.Close, test="adf")


# In[22]:


from statsmodels.graphics.tsaplots import plot_pacf


# In[24]:


diff = df.Close.diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(diff)
ax1.set_title("Difference once")
ax2.set_ylim(0, 1)
plot_pacf(diff, ax=ax2)


# In[25]:


diff = df.Close.diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(diff)
ax1.set_title("Difference once")
ax2.set_ylim(0, 1)
plot_acf(diff, ax=ax2)


# In[27]:


from statsmodels.tsa.arima_model import ARIMA


# In[28]:



# ARIMA Model
model = ARIMA(df.Close, order=(6, 1, 3))
result = model.fit(disp=0)


# In[29]:


print(result.summary())


# In[30]:


# Plot residual errors
residuals = pd.DataFrame(result.resid)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(residuals)
ax2.hist(residuals, density=True)


# In[31]:


# Actual vs Fitted
result.plot_predict(
    start=1,
    end=60,
    dynamic=False,
)


# In[33]:


n = int(len(df) * 0.8)
train = df.Close[:n]
test = df.Close[n:]


# In[34]:


print(len(train))
print(len(test))


# In[35]:


step = 30

model = ARIMA(train, order=(6, 1, 3))
result = model.fit(disp=0)

# Forecast
fc, se, conf = result.forecast(step)


# In[36]:


fc = pd.Series(fc, index=test[:step].index)
lower = pd.Series(conf[:, 0], index=test[:step].index)
upper = pd.Series(conf[:, 1], index=test[:step].index)


# In[38]:


plt.figure(figsize=(16, 8))
plt.plot(test[:step], label="actual")
plt.plot(fc, label="forecast")
plt.fill_between(lower.index, lower, upper, color="k", alpha=0.1)
plt.title("Forecast vs Actual")
plt.legend(loc="upper left")


# In[39]:


from pmdarima.arima import auto_arima


# In[41]:


model = auto_arima(
    df.Close,
    start_p=1,
    start_q=1,
    test="adf",
    max_p=6,
    max_q=6,
    m=1,  # frequency of series
    d=None,  # determine 'd'
    seasonal=False,  # no seasonality
    trace=True,
    stepwise=True,
)


# In[ ]:




