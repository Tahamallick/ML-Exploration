#!/usr/bin/env python
# coding: utf-8

# # Cross_Validation Technique

# In[4]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# # Logistic Regression
# 
# 

# In[9]:


from sklearn.model_selection import cross_val_score

l_score=cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), iris.data, iris.target,cv=3)
l_score


# In[11]:


np.average(l_score)


# # Decision Tree
# 
# 

# In[13]:


from sklearn.tree import DecisionTreeClassifier

d_score=cross_val_score(DecisionTreeClassifier(),iris.data,iris.target)


# In[14]:


d_score


# In[15]:


np.average(d_score)


# # Support Vector Machine (SVM)
# 
# 

# In[17]:


s_score=cross_val_score(SVC(),iris.data,iris.target)


# In[18]:


s_score


# In[19]:


np.average(s_score)


# # Random Forest
# 
# 

# In[50]:


r_score=cross_val_score(RandomForestClassifier(n_estimators=60),iris.data,iris.target)


# In[51]:


r_score


# In[52]:


np.average(r_score)


# In[ ]:





# In[ ]:




