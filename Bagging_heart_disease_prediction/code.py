#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\Machine-Learning-tutorial\Bagging_heart_disease_prediction\heart.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[7]:


df.describe()


# # Treat Outliers
# 

# In[8]:


df[df.Cholesterol>(df.Cholesterol.mean()+3*df.Cholesterol.std())]


# In[9]:


df.shape



# In[10]:


df1 = df[df.Cholesterol<=(df.Cholesterol.mean()+3*df.Cholesterol.std())]
df1.shape


# In[11]:


df[df.MaxHR>(df.MaxHR.mean()+3*df.MaxHR.std())]


# In[12]:


df[df.FastingBS>(df.FastingBS.mean()+3*df.FastingBS.std())]


# In[13]:


df[df.Oldpeak>(df.Oldpeak.mean()+3*df.Oldpeak.std())]


# In[14]:


df2 = df1[df1.Oldpeak<=(df1.Oldpeak.mean()+3*df1.Oldpeak.std())]
df2.shape


# In[15]:


df[df.RestingBP>(df.RestingBP.mean()+3*df.RestingBP.std())]


# In[16]:


df3 = df2[df2.RestingBP<=(df2.RestingBP.mean()+3*df2.RestingBP.std())]
df3.shape


# In[17]:


df.ChestPainType.unique()


# In[18]:


df.RestingECG.unique()


# In[19]:


df.ExerciseAngina.unique()


# In[20]:


df.ST_Slope.unique()


# # Handle text columns using label encoding and one hot encoding
# 

# In[21]:


df4 = df3.copy()
df4.ExerciseAngina.replace(
    {
        'N': 0,
        'Y': 1
    },
    inplace=True)

df4.ST_Slope.replace(
    {
        'Down': 1,
        'Flat': 2,
        'Up': 3
    },
    inplace=True
)

df4.RestingECG.replace(
    {
        'Normal': 1,
        'ST': 2,
        'LVH': 3
    },
    inplace=True)


# In[22]:


df4.head()


# In[24]:


df5 = pd.get_dummies(df4, drop_first=True)
df5=df5.astype(int)
df5.head()


# In[25]:


X = df5.drop("HeartDisease",axis='columns')
y = df5.HeartDisease

X.head()


# In[26]:


y.head()


# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20)


# In[29]:


X_train.shape


# In[30]:


X_test.shape


# # Training a model using standalone support vector machine

# In[40]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

scores = cross_val_score(SVC(), X_scaled, y, cv=5)
scores.mean()


# # Use bagging now with svm
# 
# 

# In[41]:


from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

bag_model = BaggingClassifier(base_estimator=SVC(), n_estimators=100, max_samples=0.8, random_state=0)
scores = cross_val_score(bag_model, X_scaled, y, cv=5)
mean_score = scores.mean()


# In[42]:


mean_score


# # Train a model using decision tree

# In[43]:


from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(random_state=0), X_scaled, y, cv=5)
scores.mean()


# # Use bagging now with decision tree
# 
# 

# In[45]:


bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=0), 
    n_estimators=100, 
    max_samples=0.9, 
    oob_score=True,
    random_state=0
)

scores = cross_val_score(bag_model, X_scaled, y, cv=5)
scores.mean()


# # Train a model using Random Forest which itself uses bagging underneath
# 

# In[47]:


from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
scores.mean()


# In[ ]:




