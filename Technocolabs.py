# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:54:48 2021

@author: SAMIT
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:14:32 2021

@author: SAMIT
"""

#Problem Statement
'''The data scientists at BigMart have collected 2013 sales data for 
1559 products across 10 stores in different cities. Also, certain attributes 
of each product and store have been defined. The aim of this data science 
projectis to build a predictive model and find out the sales of each 
product at a particular store. Using this model, BigMart will try to 
understand the properties of products and stores which play a key role
in increasing sales.'''




#Hypothesis Generation


#Loading Packages and Data
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir('E:\Technocolabs')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')



#Data Structure and Content
train['source']='train'
test['source']='test'
train.columns
test.columns
data = pd.concat([train, test],ignore_index=True)

'''Check which are numerical variables'''
numeric_features = data.select_dtypes(include=[np.number])
numeric_features.dtypes

train.shape, test.shape, data.shape
data.apply(lambda x: sum(x.isnull()))
data.describe()
data.apply(lambda x: len(x.unique()))





#Missing values





data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())





data['Item_Weight']=data['Item_Weight'].fillna(data.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
data.isnull().sum()
# List of item types 
item_type_list = data.Item_Type.unique().tolist()
# grouping based on item type and calculating mean of item weight
Item_Type_Means = data.groupby('Item_Type')['Item_Weight'].mean()
# Mapiing Item weight to item type mean
for i in item_type_list:
    dic = {i:Item_Type_Means[i]}
    s = data.Item_Type.map(dic)
    data.Item_Weight = data.Item_Weight.combine_first(s)
    

Item_Type_Means = data.groupby('Item_Type')['Item_Weight'].mean() 
# Checking if Imputation was successful
data.isnull().sum()





def impute_size(cols):
    size = cols[0]
    ot_type = cols[1]
    if pd.isnull(size):
        if ot_type == "Supermarket Type1":
            return "Small"
        elif ot_type == "Supermarket Type2":
            return "Medium"
        elif ot_type == "Grocery Store":
            return "Small"
        elif ot_type == "Supermarket Type3":
            return "Medium"
    return size  
data["Outlet_Size"] = data[["Outlet_Size","Outlet_Type"]].apply(impute_size, axis = 1)
data.isnull().sum()





#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
missing_values = (data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(missing_values))
data.loc[missing_values,'Item_Visibility'] = data.loc[missing_values,'Item_Identifier'].apply(lambda x: visibility_avg.at[x, 'Item_Visibility'])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))


data.apply(lambda x: sum(x.isnull()))


data['Item_Fat_Content'] =data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
data['Item_Fat_Content'].value_counts()






data.index = data['Outlet_Establishment_Year']
data.index

df = data.loc[:,['Item_Outlet_Sales']]
df.head(2)
data['Outlet_Years'] = 2009 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()




#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                         'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()



    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data.apply(LabelEncoder().fit_transform)

# one hot encoding

data = pd.get_dummies(data, columns=['Item_MRP','Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined'])
data.head()
data.columns

# Exporting Data

import warnings
warnings.filterwarnings('ignore')

# splitting the dataset into train and test

train = data.iloc[:8523,:]
test = data.iloc[8523:,:]
# Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'], axis = 1, inplace=True)
train.drop(['source'], axis = 1, inplace = True)

# Export files as modified versions
train.to_csv("C:/Users/Samit/Desktop/train_modified.csv", index = False)
test.to_csv("C:/Users/Samit/Desktop/test_modified.csv", index = False)


#Model Building
# Reading modified data 
os.chdir('C:/Users/Samit/Desktop/Internship')
train2 = pd.read_csv("train_modified.csv")
test2 = pd.read_csv("test_modified.csv")
import statsmodels.api as sm
train2.columns
x = train2[[ 'Item_MRP', 'Item_Fat_Content_Low Fat',
       'Item_Fat_Content_Regular',  'Outlet_Type_Grocery Store', 'Outlet_Type_Supermarket Type1',
       'Outlet_Type_Supermarket Type2', 'Outlet_Type_Supermarket Type3']]
y = train2['Item_Outlet_Sales']
x.info()

x.shape, y.shape

 
y = np.array(y).reshape(8523,1)
#x = sm.add_constant(x)
lm = sm.OLS(y,x)
results = lm.fit()
results.summary()

import pickle
filename = 'new_model.pkl'
pickle.dump(results, open(filename, 'wb'))






from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
LinearRegression()

# Predicting the test set results

y_pred = regressor.predict(x)
y_pred
# Measuring Accuracy

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import metrics
lr_accuracy = round(regressor.score(x,y) * 100,2)
lr_accuracy
r2_score(y, regressor.predict(x))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, regressor.predict(x))))









#Random Forest Model
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=50, n_jobs=4)
regressor.fit(x, y)
RandomForestRegressor(max_depth=6, min_samples_leaf=50, n_jobs=4)

# Predicting the test set results

y_pred = regressor.predict(x)
y_pred
rf_accuracy = round(regressor.score(x,y),2)
rf_accuracy
r2_score(y, regressor.predict(x))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, regressor.predict(x))))





#Decision tree
from sklearn.tree import DecisionTreeRegressor
Dtree = DecisionTreeRegressor(max_depth=3)
Dtree.fit(x, y)
y_pred = Dtree.predict(x)
y_pred
rf_accuracy = round(Dtree.score(x,y),2)
rf_accuracy
r2_score(y, Dtree.predict(x))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, Dtree.predict(x))))




#XgBoost Regressor

from sklearn.ensemble import GradientBoostingRegressor

xgb = GradientBoostingRegressor()
xgb.fit(x, y)

# predicting the test set results
y_pred = xgb.predict(x)
print(y_pred)

rf_accuracy = round(xgb.score(x,y),2)
rf_accuracy
r2_score(y,xgb.predict(x))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, xgb.predict(x))))



import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(xgb, pickle_out) 
pickle_out.close()


pip install -q pyngrok

pip install -q streamlit

pip install -q streamlit_ace





