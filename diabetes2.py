#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('C:/Users/uppal/Desktop/Diabetes prediction/Diabetes.csv')



# In[7]:


data.head()


# In[10]:


data.isnull().any()


# In[11]:


zero_values = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for col in zero_values:
    data[col]= data[col].replace(0,np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN,mean)


# In[12]:


X = data.iloc[:,0:8]


# In[13]:


y = data.iloc[:,8]


# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[18]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')


# In[19]:


classifier.fit(X_train,y_train)


# In[20]:



y_pred = classifier.predict(X_test)


# In[26]:


conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(f1_score(y_test,y_pred, pos_label='pos'))


# In[23]:


print(accuracy_score(y_test,y_pred))


# In[ ]:




