import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.chdir('C:/Users/HP/Desktop/Technocolabs')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

#EDA
train.shape, test.shape
train.columns
test.columns

"""We need to predict Item_Outlet_Sales for given test data
lets first merge the train and test data for Exploratory Data Analysis"""

train['source'] = 'train'
test['source'] = 'test'
test['Item_Outlet_Sales'] = 0
data = pd.concat([train, test], sort = False)
print(train.shape, test.shape, data.shape)

data.head()


#HANDLING MISSING VALUES
""" get the number of missing data points per column"""
missing_values_count = train.isnull().sum()


"""replace all NA's the value that comes directly after it in the same column, 
 then replace all the remaining na's with 0"""
train.fillna(method='bfill', axis=0).fillna(0)

train.describe()



#Univariate analysis
'''Item_Outlet_Sales'''
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
sns.kdeplot(data=train['Item_Outlet_Sales'], shade=True)

'''Check which are numerical variables'''
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

'''Correlation between Numerical Predictors and Target variable'''
corr =numeric_features.corr()
corr
print(corr['Item_Outlet_Sales'].sort_values(ascending=False))
#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);



'''Check categorical variables'''
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")



 '''Item_Fat_Content'''
print ('Original Categories:')
print (train['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
train['Item_Fat_Content'] =train['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (train['Item_Fat_Content'].value_counts())


sns.countplot(train.Item_Fat_Content)


'''Item_Type'''
sns.countplot(train.Item_Type)
plt.xticks(rotation=90)

'''Outlet_Size'''
sns.countplot(train.Outlet_Size)

'''Outlet_Location_Type'''
sns.countplot(train.Outlet_Location_Type)


'''Outlet_Type'''
sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)



#Bivariate analysis
