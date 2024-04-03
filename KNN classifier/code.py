#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:


dir(digits)


# In[3]:


df = pd.DataFrame(digits.data,digits.target)
df.head()


# In[4]:


df['target'] = digits.target
df.head(20)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.2, random_state=10)


# # Create KNN classifier
# 
# 

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
X_test_np = X_test.to_numpy()

knn = KNeighborsClassifier(n_neighbors=3)


# In[26]:


len(X_train)


# In[27]:


len(X_test)


# In[28]:


knn.fit(X_train, y_train)


# In[30]:


accuracy = knn.score(X_test_np, y_test)
print("Accuracy:", accuracy)


# In[32]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test_np)
cm = confusion_matrix(y_test, y_pred)
cm


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[34]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:




