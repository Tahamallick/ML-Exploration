#!/usr/bin/env python
# coding: utf-8

# #                                       Random Forest Algorithim

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


dir(iris)


# In[4]:


iris.feature_names


# In[5]:


iris.target_names


# In[6]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[7]:


df['target'] = iris.target
df.head()


# In[9]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head(150)


# # Train and the model and prediction
# 
# 
# 

# In[14]:


X = df.drop(['target', 'flower_name'], axis='columns')

y = df.target


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)


# In[17]:


model.score(X_test,y_test)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, y_train)


# In[19]:


model.score(X_test,y_test)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, y_train)


# In[21]:


model.score(X_test,y_test)


# In[ ]:




