#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[9]:


dataset = pd.read_csv('C:/path/to/your/directory/canada_per_capita_income.csv')



# In[10]:


dataset = pd.read_csv('C:/Users/Taha/Desktop/canada_per_capita_income.csv')


# In[11]:


dataset = pd.read_csv(r'C:\Users\Taha\Desktop\canada_per_capita_income.csv')


# In[14]:


df = pd.read_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\predicting the canada income in 2020\canada_per_capita_income.csv')


# In[15]:


df


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df.year, df['per capita income (US$)'], color='red', marker='+')


# In[25]:


new_df = df.drop('per capita income (US$)',axis='columns')
new_df


# In[30]:


per_capita_income=df['per capita income (US$)']


# In[31]:


per_capita_income


# In[32]:


from sklearn import linear_model


# In[34]:


reg=linear_model.LinearRegression()
reg.fit(new_df,per_capita_income)


# In[35]:


reg.predict([[2020]])


# In[37]:


predicted_income_2020 = reg.predict([[2020]])


# In[38]:


prediction_df = pd.DataFrame({'Year': [2020], 'Predicted Per Capita Income (US$)': predicted_income_2020})


# In[39]:


print(prediction_df)


# In[42]:


prediction_df.to_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\predicting the canada income in 2020\predicted_per_capita_income_2020.csv', index=False)


# In[ ]:




