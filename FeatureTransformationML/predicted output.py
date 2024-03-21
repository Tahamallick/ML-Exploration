#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


df = pd.read_csv(r"C:/Users/Taha/Desktop/machinelearning tutorial/Machine-Learning-tutorial/FeatureTransformationML/carprices.csv")


# In[6]:


df


# In[19]:


dummies = pd.get_dummies(df['Car Model'], dtype=int)


# In[20]:


dummies


# In[21]:


merged = pd.concat([df,dummies],axis='columns')
merged


# In[22]:


final = merged.drop(['Car Model'], axis='columns')


# In[23]:


final


# In[25]:


final = final.drop(['Mercedez Benz C class'], axis='columns')


# In[26]:


final


# In[27]:


X = final.drop('Sell Price($)', axis='columns')
X


# In[30]:


y = final['Sell Price($)']


# In[31]:


y


# In[32]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[33]:


model.fit(X,y)


# In[34]:


model.predict(X)


# In[35]:


model.score(X,y)


# #Price of mercedez benz that is 4 yr old with mileage 45000
# 

# In[40]:


predicted_price = model.predict([[45000, 4, 0, 0]])


# In[41]:


predicted_price


# #Price of BMW X5 that is 7 yr old with mileage 86000
# 
# 

# In[38]:


predicted_price1 = model.predict([[86000, 7, 0, 1]])                  


# In[39]:


predicted_price1


# In[ ]:




