#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[13]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)


# In[14]:


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
plt(X_test)


# In[19]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
print(X_test)


# In[ ]:




