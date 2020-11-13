#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Trip Prediction

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('taxifare.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# # Feature Engineering with respect to Datetime

# In[6]:


import datetime


# In[7]:


pd.to_datetime(df['pickup_datetime'])


# In[8]:


df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])-datetime.timedelta(hours=4)


# In[9]:


df.info()


# In[10]:


df.head()


# In[11]:


df['pickup_datetime'].dt.month


# In[12]:


df['Year']=df['pickup_datetime'].dt.year
df['Month']=df['pickup_datetime'].dt.month
df['Day']=df['pickup_datetime'].dt.day
df['Hours']=df['pickup_datetime'].dt.hour
df['Minutes']=df['pickup_datetime'].dt.minute


# In[13]:


df.head()


# In[14]:


import numpy as np


# In[15]:


df['mornight']=np.where(df['Hours']<12,0,1)


# In[16]:


df.head()


# In[17]:


df.drop('pickup_datetime',axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


df['fare_class'].unique()


# In[20]:


### https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html#:~:text=The%20Haversine%20(or%20great%20circle,the%20data%20must%20be%202
from sklearn.metrics.pairwise import haversine_distances
from math import radians
newdelhi = [28.6139, 77.2090]
bangalore = [12.9716, 77.5946]


# In[21]:


newdelhi_in_radians = [radians(_) for _ in newdelhi]
bangalore_in_radians = [radians(_) for _ in bangalore]


# In[22]:


result = haversine_distances([newdelhi_in_radians, bangalore_in_radians])


# In[23]:


result*6371


# In[24]:


np.radians(df['dropoff_latitude']-df["pickup_latitude"])


# # Calculating The Haversine Distance

# In[25]:


###https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points


def haversine(df):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1= np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    #### Based on the formula  x1=drop_lat,x2=dropoff_long 
    dlat = np.radians(df['dropoff_latitude']-df["pickup_latitude"])
    dlong = np.radians(df["dropoff_longitude"]-df["pickup_longitude"])
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# In[26]:


df['Total distance']=haversine(df)


# In[27]:


df.head()


# In[28]:


df.drop(["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"],axis=1,inplace=True)


# In[29]:


df.head()


# In[30]:


X=df.iloc[:,1:]
y=df.iloc[:,0]


# In[31]:


### Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[32]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()


# In[50]:


X.head()


# In[51]:


y.head()


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


# In[36]:





# In[52]:



from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 800, num = 8)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[53]:



# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[54]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[55]:


# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[56]:


rf_random.fit(X_train,y_train)


# In[57]:


y_pred=rf_random.predict(X_test)


# In[58]:


import seaborn as sns

sns.distplot(y_test-y_pred)


# In[59]:


plt.scatter(y_test,y_pred)


# In[60]:


from sklearn import metrics
print('R square:', np.sqrt(metrics.r2_score(y_test, y_pred)))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




