#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression: Multiclass Classification
# 

# In[13]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[14]:


iris


# In[3]:


dir(iris)


# In[11]:


iris.feature_names[:4]


# In[15]:


iris.data[5]


# #Create and train logistic regression model

# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[17]:


from sklearn.model_selection import train_test_split


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.3)


# In[76]:


model.fit(X_train, y_train)


# #Accuracy of our model
# 

# In[77]:


model.score(X_test, y_test)


# In[78]:


model.predict(iris.data[0:7])


# In[79]:


y_predicted = model.predict(X_test)


# In[80]:


y_predicted


# In[81]:


flower_types = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}


# In[82]:


predictions = model.predict(iris.data[:150])



# #predictions to flower types

# In[83]:


predicted_flowers = [flower_types[prediction] for prediction in predictions]
flower_types = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}


# In[84]:


for i, sample in enumerate(iris.data[:150]):
    print("Sample {}: {}".format(i+1, predicted_flowers[i]))


# #Confusion Matrix
# 

# In[85]:


y_predicted = model.predict(X_test)


# In[86]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[87]:


import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




