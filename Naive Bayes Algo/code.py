#!/usr/bin/env python
# coding: utf-8

# # Naive bayes

# In[1]:


from sklearn import datasets
wine = datasets.load_wine()


# In[2]:


dir(wine)


# In[3]:


wine.data[0:2]


# In[4]:


wine.feature_names


# In[5]:


wine.target_names


# In[6]:


wine.target[0:2]


# In[7]:


import pandas as pd
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df.head()


# In[8]:


df['target'] = wine.target
df[50:70]


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=100)


# In[10]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(X_train,y_train)


# In[11]:


model.score(X_test,y_test)


# In[12]:


model = MultinomialNB()
model.fit(X_train,y_train)


# In[13]:


model.score(X_test,y_test)


# In[ ]:




