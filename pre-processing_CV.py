#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")


# In[7]:


train = train.drop(["Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned", "Awards/honours", "Avg quality score"], axis = 1)
test = test.drop(["Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned", "Awards/honours", "Avg quality score"], axis = 1)


# In[13]:


train


# In[22]:


for i in train.columns:
    if i== "File name":
        continue
    for j in range(0, len(train[i])):
        train.loc[j, i] = float(train[i][j])
        # print(type(train[i][j]))


# In[23]:


for i in train.columns:
    # train[i] = float(train[i])
    if i == "File name":
        continue
    train[i] = (train[i] - train[i].min()) / (train[i].max() - train[i].min())


# In[24]:


for i in test.columns:
    if i== "File name":
        continue
    for j in range(0, len(test[i])):
        test.loc[j, i] = float(test[i][j])
        # print(type(train[i][j]))


# In[25]:


for i in test.columns:
    if i == "File name":
        continue
    test[i] = (test[i] - test[i].min()) / (test[i].max() - test[i].min())


# In[ ]:





# In[ ]:




