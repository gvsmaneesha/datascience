
# coding: utf-8

# In[ ]:


#Load libraries
import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


#Set working directory
os.chdir("E:\Others\Edwisor\ContentRevamp\MarketingCampaign")


# In[ ]:


#Load data
df = pd.read_csv("df.csv")


# In[ ]:


#Divide data into train and test
train, test = train_test_split(df, test_size=0.2)


# ## Decision Tree

# In[ ]:


#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9], train.iloc[:,9])

#Apply model on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:9])


# In[ ]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

MAPE(test.iloc[:,9], predictions_DT)


# In[ ]:


#MAPE: 12.35


