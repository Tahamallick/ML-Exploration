#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\Machine-Learning-tutorial\Talent Trendz\HR_comma_sep.csv')


# In[3]:


df


# In[5]:


df.head()


# # Data exploration and visualization
# 

# In[7]:


left = df[df.left==1]
left.shape


# In[8]:


retained=df[df.left==0]


# In[9]:


retained.shape


# #Average numbers for all columns
# 
# 

# In[14]:


new_df=df.drop(columns=['Department','salary'])


# In[15]:


new_df.groupby('left').mean()


# #Impact of salary on employee retention
# 
# 

# In[16]:


import matplotlib.pyplot as plt


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


pd.crosstab(df.salary,df.left).plot(kind='bar')


# #Department wise employee retention rate
# 
# 

# In[19]:


pd.crosstab(df.Department,df.left).plot(kind='bar')


# #conclude from the above analysis we are ussing the variables 
# #**Satisfaction Level****Average Monthly Hours****Promotion Last 5 Years****Salary** as an independent variable

# #converting salary into dummy variable

# In[20]:


subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()


# In[26]:


salary_dummies = pd.get_dummies(subdf.salary, prefix="salary", dtype=int)



# In[27]:


df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')


# In[28]:


df_with_dummies


# In[29]:


df_with_dummies.drop('salary',axis='columns',inplace=True)
df_with_dummies.head()


# In[30]:


X = df_with_dummies
X.head()


# In[31]:


y=df.left


# # Logistic Regression Model

# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5)


# In[42]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[43]:


model.fit(X_train, y_train)


# In[44]:


model.predict(X_test)


# # Accuracy of the model
# 
# 

# In[46]:


model.score(X_test,y_test)


# In[ ]:




