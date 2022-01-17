#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv("recipes_muffins_cupcakes.csv")


# In[4]:


dataset


# In[12]:


x=dataset.iloc[:,1:]


# In[13]:


x


# In[15]:


y=dataset.Type


# In[42]:


y


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2)


# In[45]:


X_train


# In[46]:


y_train


# In[47]:


from sklearn import svm
model = svm.SVC(kernel='linear')
model.fit(X_train,y_train)


# In[35]:


model


# In[48]:


y_pred=model.predict(X_test)


# In[49]:


y_pred


# In[52]:


from sklearn.metrics import accuracy_score


# In[53]:


score=accuracy_score(y_test,y_pred)


# In[55]:


score*100


# In[58]:


from sklearn.metrics import confusion_matrix


# In[59]:


score1=confusion_matrix(y_test,y_pred)


# In[60]:


score1


# In[64]:


from sklearn.metrics import classification_report


# In[65]:


score3=classification_report(y_test,y_pred)


# In[66]:


score3


# In[ ]:




