#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\Machine-Learning-tutorial\pca-heart-prediction\heart.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


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


df3=df2[df2.RestingBP<=(df2.RestingBP.mean()+3*df.RestingBP.std())]


# In[17]:


df3.shape


# In[18]:


df.ChestPainType.unique()


# In[19]:


df.RestingECG.unique()



# In[20]:


df.ExerciseAngina.unique()


# In[21]:


df.ST_Slope.unique()


# In[22]:


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


# In[24]:


df4.head()


# In[26]:


df5 = pd.get_dummies(df4, drop_first=True,dtype=int)
df5.head()


# In[27]:


X = df5.drop("HeartDisease",axis='columns')
y = df5.HeartDisease

X.head()


# In[28]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)


# In[30]:


X_train.shape


# In[31]:


X_test.shape


# In[32]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)


# # Use PCA to reduce dimensions
# 

# In[33]:


X


# In[37]:


from sklearn.decomposition import PCA

pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
X_pca


# In[38]:


X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)


# In[39]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train_pca, y_train)
model_rf.score(X_test_pca, y_test)


# In[ ]:




