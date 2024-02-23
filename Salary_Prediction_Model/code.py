#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[4]:


df = pd.read_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\multi-value prediction\hiring.csv')
df


# In[7]:


df['test_score(out of 10)'].median()


# In[8]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())
df


# In[9]:


reg = linear_model.LinearRegression()
reg.fit(df.drop('salary($)',axis='columns'),df['salary($)'])


# In[10]:


from word2number import w2n


# In[11]:


get_ipython().system('pip install word2number')


# In[12]:


from word2number import w2n



# In[13]:


word_to_num_mapping = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
}


# In[17]:


def convert_to_number(word):
    return word_to_num_mapping.get(word, None)


# In[18]:


df['experience_numeric'] = df['experience'].map(convert_to_number)



# In[19]:


df


# In[20]:


df.experience_numeric.median()


# In[21]:


df['experience_numeric'] = df['experience_numeric'].fillna(df['experience_numeric'].median())


# In[22]:


df



# In[23]:


df.drop(columns=['experience'], inplace=True)


# In[24]:


df


# In[25]:


reg = linear_model.LinearRegression()
reg.fit(df.drop('salary($)',axis='columns'),df['salary($)'])


# In[61]:


new_data_points = pd.DataFrame({ 'test_score(out of 10)': [9, 10], 'interview_score(out of 10)': [6, 10],'experience_numeric': [2, 12]})

predicted_salaries = reg.predict(new_data_points)

predicted_salaries


# In[64]:


new_data_points['salary($)'] = predicted_salaries
df_with_predictions = pd.concat([df, new_data_points], ignore_index=True)




# In[68]:


df_with_predictions.to_csv(r'C:\Users\Taha\Desktop\machinelearning tutorial\multi-value prediction\predicted_model_processed.csv', index=False)


# In[44]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].astype(int)


# In[47]:


df['experience_numeric'] = df['experience_numeric'].astype(int)


# In[66]:


df


# In[67]:


df_with_predictions


# In[ ]:




