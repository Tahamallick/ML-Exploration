#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classification
# 

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv(r"C:\Users\Taha\Desktop\machinelearning tutorial\Machine-Learning-tutorial\survival-prediction-model\titanic.csv")
df.head()


# In[4]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)


# In[5]:


df.head()


# In[8]:


inputs = df.drop('Survived',axis='columns')

target = df['Survived']


# In[9]:


inputs


# In[10]:


target


# In[11]:


inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})


# In[12]:


inputs.Age[:10]


# In[13]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[14]:


inputs


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[17]:


len(X_train)


# In[18]:


len(X_test)



# In[19]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[20]:


model.fit(X_train,y_train)


# In[21]:


model.score(X_test,y_test)


# In[ ]:




