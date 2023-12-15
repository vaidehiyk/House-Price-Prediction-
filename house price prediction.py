#!/usr/bin/env python
# coding: utf-8

# In[ ]:


house price predection


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=read_csv('Downloads//kc_house_data.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values,y=data.long.values,size=10)
plt.ylabel('Longitude',fontsize=12)
plt.xlabel('latitude',fontsize=12)
plt.show()
sns.despine


# In[ ]:


plt.scatter(data.price,data.sqft_living)
plt.title('price vs sqfoot')


# In[ ]:


plt.scatter(data.price,data.long)
plt.title('price vs location of area')


# In[ ]:


plt.scatter(data.bedroom,data.price)
plt.tite('Bedroom and Price')
plt.xlabel('Bedroom')
plt.ylabel('Price')
plt.show()
sns.despine


# In[ ]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[ ]:


plt.scatter(data.waterfront,data.price)
plt.title('waterfront vs price(0=no waterfront)')


# In[ ]:


plt.scatter(data.floors,data.price)


# In[ ]:


plt.scatter(data.condition,data.price)


# In[ ]:


plt.scatter(data.zipcode,data.price)
plt.title('which is the pricey location by zipcode')


# In[ ]:


from sklearn.liner_model import LinearRegression


# In[ ]:


reg=LinearRegression()


# In[ ]:


labels=data['price']
conv_dates=[1 if values==2014 else 0 for values in data.date]
data['date']=conv_dates


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train1,labels,test_size=0.10,random_state=2)


# In[ ]:


reg.fit(x_train,y_train)


# In[ ]:


reg.score(x_test,y_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
est=GradientBoostingRegressor(n_estimators=2000,max_depth=1).fit(x,y)
for pred in est.staged_predict(x):
    plt.plot(x[:,0],pred,color-'r',aplha=0.1)


# In[ ]:


from sklearn import ensemble
clf=ensemble.GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split=2,
                                      learning_rate=0.1,loss='ls')


# In[ ]:


clf.fit(x_train,y_train)


# In[ ]:


clf.score(x_test,y_test)

