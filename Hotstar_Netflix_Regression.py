#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("mediacompany.csv")
df.drop(["Unnamed: 7"], axis = 1, inplace = True)


# In[3]:


df["Date"] = pd.to_datetime(df.Date).dt.date
df.head()


# In[4]:


from datetime import date

d0 = date(2017, 2, 28)
d1 = df.Date
delta = d1 - d0
df["day"] = delta
df.head()


# In[5]:


df["day"] = df.day.astype(str)
df["day"] = df.day.map(lambda x: x[0:2])
df["day"] = df.day.astype(int)


# In[6]:


df.head()


# In[76]:


df.plot.line("day", "Views_show")
plt.grid()


# In[14]:


X = df[["Visitors", "Character_A"]]
Y = df["Views_show"]


# In[18]:


import statsmodels.api as am
X = sm.add_constant(X)
lm_1 = sm.OLS(Y,X).fit()
lm_1.summary()


# In[21]:


df["weekday"] = (df["day"]+3)%7
df.weekday.replace(0, 7, inplace = True)
df.weekday.astype(int)
df.head()


# In[22]:


df.weekday.replace(2, 0, inplace = True)
df.weekday.replace(3, 0, inplace = True)
df.weekday.replace(4, 0, inplace = True)
df.weekday.replace(5, 0, inplace = True)
df.weekday.replace(6, 0, inplace = True)
df.weekday.replace(7, 1, inplace = True)


# In[25]:


df.rename(columns={"weekday":"weekend"}, inplace=True)


# In[26]:


df.head()


# In[30]:


X = df[["Visitors", "Character_A", "weekend"]]

X = sm.add_constant(X)
lm_2 = sm.OLS(Y, X).fit()
lm_2.summary()


# In[39]:


X = df[["Character_A", "weekend", "Ad_impression"]]

X = sm.add_constant(X)
lm_3 = sm.OLS(Y, X).fit()
lm_3.summary()


# In[41]:


pred = lm_3.predict(X)


# In[75]:


plt.plot(df.day, Y, color = "red", label = "Actual")
plt.plot(df.day, pred, color = "blue", label = "Predicted")
plt.xlabel("Days")
plt.ylabel("Viewers")
plt.grid()


# In[71]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
model = LinearRegression()
model.fit(X_train, Y_train)


# In[72]:


model.score(X_test, Y_test)


# In[79]:


plt.plot(df.day, (Y-pred), color = "red", label = "Error")
plt.grid()


# In[ ]:




