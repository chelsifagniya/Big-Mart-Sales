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
os.chdir('C:/Users/HP/Desktop/CHELSI/Technocolabs')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')



#Data Structure and Content
train['source']='train'
test['source']='test'
train.columns
test.columns
test['Item_Outlet_Sales'] = 0
data = pd.concat([train, test],ignore_index=True)

'''Check which are numerical variables'''
numeric_features = data.select_dtypes(include=[np.number])
numeric_features.dtypes

train.shape, test.shape, data.shape
data.apply(lambda x: sum(x.isnull()))
data.describe()
data.apply(lambda x: len(x.unique()))



#EDA
pip install sweetviz
import sweetviz
import pandas as pd
my_report = sweetviz.compare([train, "Train"], [test, "Test"], 'Item_Outlet_Sales')
my_report.show_html("Report.html")

#Univariate Analysis

'''Item_Outlet_Sales i.e target variable'''
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")


sns.distplot(a=train['Item_Outlet_Sales'], kde=False)

sns.kdeplot(data=train['Item_Weight'], shade=True)




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



sns.countplot(data.Item_Fat_Content)


'''Item_Type'''
sns.countplot(data.Item_Type)
plt.xticks(rotation=90)

'''Outlet_Size'''
sns.countplot(data.Outlet_Size)

'''Outlet_Location_Type'''
sns.countplot(data.Outlet_Location_Type)


'''Outlet_Type'''
sns.countplot(data.Outlet_Type)
plt.xticks(rotation=90)

#Bivariate analysis
 '''Item_Weight and Item_Outlet_Sales analysis'''
plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(data.Item_Weight, data["Item_Outlet_Sales"],'.', alpha = 0.3)


'''Item_Visibility and Item_Outlet_Sales analysis'''
plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(data.Item_Visibility, data["Item_Outlet_Sales"],'.', alpha = 0.3)



#Missing values
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






#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                         'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()



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






    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data.apply(LabelEncoder().fit_transform)

# one hot encoding

data = pd.get_dummies(data)

print(data.shape)


x=data.drop(['Item_Outlet_Sales'],axis=1)
y=data['Item_Outlet_Sales']


print(x.shape)
print(y.shape)



# splitting the dataset into train and test

train = data.iloc[:8523,:]
test = data.iloc[8523:,:]

print(train.shape)
print(test.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(x_train, y_train)

# predicting the  test set results
y_pred = model.predict(x_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print('RMSE :', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

#Decision tree
from sklearn.tree import DecisionTreeRegressor
Dtree = DecisionTreeRegressor(max_depth=3)
Dtree.fit(x_train, y_train)
y_predicted = Dtree.predict(x_test)
y_predicted
mse = mean_squared_error(y_test,y_predicted)
print('RMSE :', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test,y_predicted))




#AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
model= AdaBoostRegressor(n_estimators = 100)
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# RMSE
mse = mean_squared_error(y_test, y_pred)
print("RMSE :", np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))





#XgBoost Regressor

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)
print(y_pred)

# Calculating the root mean squared error
print("RMSE :", np.sqrt(((y_test - y_pred)**2).sum()/len(y_test)))



#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 100 , n_jobs = -1)
model.fit(x_train, y_train)

# predicting the  test set results
y_pred = model.predict(x_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print("RMSE :",np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))







