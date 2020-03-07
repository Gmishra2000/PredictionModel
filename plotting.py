# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:02:25 2020

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data_income=pd.read_csv('income.csv')
data=data_income.copy()

"""
#Exploratory data analysis
#1.Getting to know data

"""
#to check the data type
print(data.info())

#check for missing values
data.isnull()
print('Data columns with null values:\n',data.isnull().sum())
#No missing value

#summary of numerical value
summary_num=data.describe()
print(summary_num)

#summary of categorical value o reprsents object
summary_cate=data.describe(include="O")
print(summary_cate)

#Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#checking for uniwue classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

#There exists '?' instead of nan
data=pd.read_csv('income.csv',na_values=[" ?"])


#data pre-processing
data.isnull().sum()

missing=data[data.isnull().any(axis=1)]
#axis=1 => to consider at leat one column value is missing

missing.isnull().sum()


data2=data.dropna(axis=0)

#Relation between independent variables

correlation=data2.corr()
"""
correlation lies between -1 and 1.
If it is closer to 1 it has strong relation between values.
If it is closer to 0 it has no relation  or less relation between values.

"""
data2.columns

#Gender proportion table
gender=pd.crosstab(index=data2["gender"],
                   columns='count',
                   normalize=True)
print(gender)

#gender vs salary status:
gender_salstat=pd.crosstab(index=data2['gender'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index')
print(gender_salstat)

#Frequency distribution of salary status

SalStat=sns.countplot(data2['SalStat'])

#Histogram Of Age
sns.distplot(data2['age'],bins=10,kde=False)

#################Box Plot- Age  vs salary status##################
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

#Education vs Salary status
Education_salstat=pd.crosstab(index=data2['EdType'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index')
print(Education_salstat)
Education=sns.countplot(data2['capitalgain'])


 # #################################Logistic Regression##################################
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
  
#The method which we have used is integer encoding 
#We are using concept of one hot Encoding
  
new_data=pd.get_dummies(data2,drop_first=True)
  
#storing the column names
columns_list=list(new_data.columns)
print(columns_list)
       
       
#seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)
       
#Storing the output values in y
y=new_data['SalStat'].values
print(y)
       
#Storing the values from input features
x=new_data[features].values
print(x)
       
#Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)
       
#make an instance of the model
logistic=LogisticRegression()
       
       
       
#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
       
#prediction from test data
prediction=logistic.predict(test_x)
print(prediction)
       
#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)
       
#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
       
       
#printing the misclassified values from prediction 
print('Misclassified samples: %d' % (test_y != prediction).sum())
       
       
       
####################Logistic regression- Removing insignificant variables####################### 
#Reindexing  the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
       
cols=['gender','nativecountry','race','JobType']
new_data=data.drop(cols,axis=1)
       
new_data=pd.get_dummies(new_data,drop_first=True)
 
#Starting the column names
columns_list=list(new_data.columns)
print(columns_list)
 
#Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)
 
#Starting the output values in y
y=new_data['SalStat'].values 
print(y)
 
#sStarting the input values from input features
x=new_data[features].values     
print(x)
 
#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)
  
#make an instance of the model
logistic=LogisticRegression()
       
#Fitting the values for x and y
logistic.fit(train_x,train_y)   
       
#prediction from test data
prediction=logistic.predict(test_x)
       
#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
       
#printing the misclassified values from prediction 
print('Misclassified samples:%d' % (test_y !=prediction).sum())
       
############KNN######################
       
#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier
       
#import library for plotting
import matplotlib.pyplot as plt
       
#Storing the Knearest neighbour classifier
KNN_classifier=KNeighborsClassifier(n_neighbors =5)
       
#Fitting the values for X and Y
KNN_classifier.fit(train_x,train_y)
       
#predicting the test values with model
prediction=KNN_classifier.predict(test_x)
       
#performance metrix check
confusion_matrix=confusion_matrix(test_y, prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)
       
#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
      
#printing the misclassified values from prediction 
print('Misclassified samples:%d' (test_y !=prediction).sum())
       
       """
       Effect of k value on classifier
       """
Misclassified_sample=[]
# calculating errors for k values between 1 and 20
      for i in range(1,20):
          knn=KNeighborsClassifier(n_neighbors=i)
          knn.fit(train_x,train_y)
          pred_i=knn.predict(test_x)
          Misclassified_sample.append((test_y !=pred_i).sum())
print(Misclassified_sample)
     
     #End of script#
       
 