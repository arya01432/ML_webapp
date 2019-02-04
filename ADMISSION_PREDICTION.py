#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[103]:


df=pd.read_csv("F://Admission_Predict.csv")
df.head()


# In[104]:


print(df.shape)


# In[105]:


X=df[['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']]
X


# In[106]:


y=df['Chance_of_admit'].values


# In[107]:


X=np.asarray(X)
y=np.asarray(y)


# In[110]:


from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
X[:2]


# In[112]:


plt.figure(figsize=(20,10))
plt.plot(X,y,'bo')
plt.show()


# In[114]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[115]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)


# In[116]:


LR=LR.fit(X_train,y_train)
y_testpred=LR.predict(X_test)


# In[118]:


from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_test,y_testpred)))


# In[121]:


print(LR.score(X_train,y_train))


# In[127]:


plt.figure(figsize=(20,10))
plt.plot(X_test,y_test,'bo',color='green')
plt.plot(X_test,y_testpred,'bo',color='red')
plt.show()

