#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine Tutorial Using Python Sklearn
# 

# In[2]:


import pandas as pd
from sklearn.datasets import load_digits
digits= load_digits()


# In[3]:


dir(digits)


# In[7]:


digits.data


# In[9]:


digits.target_names


# In[17]:


df = pd.DataFrame(digits.data,digits.target)
df.head()


# In[19]:


df['target'] = digits.target
df.head(25)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)


# # Using RBF kernel
# 
# 

# In[22]:


from sklearn.svm import SVC
rbf_model = SVC(kernel='rbf')


# In[23]:


len(X_train)


# In[24]:


len(X_test)


# In[25]:


rbf_model.fit(X_train, y_train)


# In[26]:


rbf_model.score(X_test,y_test)


# # Using linear kernel

# In[27]:


from sklearn.svm import SVC
linear_model = SVC(kernel='linear')


# In[28]:


linear_model.fit(X_train,y_train)


# In[29]:


linear_model.score(X_test,y_test)


# In[ ]:




