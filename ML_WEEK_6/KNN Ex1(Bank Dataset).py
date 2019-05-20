#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[30]:


# read file
df_original=pd.read_csv("bank.csv",delimiter=";")

df =df_original
df.head()


# In[31]:


df.dtypes


# In[32]:


df.isnull().sum()


# In[33]:


df.duplicated().sum()


# In[34]:


df.info()


# In[35]:


df.describe()


# In[36]:


df.columns


# In[37]:


df.drop(['marital','contact','month'], axis=1,inplace=True)


# In[38]:


df.head()


# In[39]:


df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[40]:


df = pd.get_dummies(df)
df.head()    


# In[41]:


df.shape


# In[42]:


# sb.boxplot(data=df)
# check for ouliers
df.boxplot(rot=45, figsize=(20,5))


# In[43]:


df.shape


# In[44]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df


# In[45]:


df = Feature_Scaling(df)
df.head()


# In[51]:


def Split(data):
    train=int(0.70*len(data))
    train_data=df.head(train)
#     print(train)
    test=int(0.30*len(data))
    test_data=df.tail(test)
   
    return train,test
train,test = Split(df)


# In[56]:


# Separating the output and the parameters data frame
def separate(df):
    output = df.y
    return df.drop('y', axis=1), output

x_train_data,y_train_data = separate(train_data)
x_test_data,y_test_data = separate(test_data)

# calling method by class object
    
x_train_data = np.array(x_train_data[:2000])
y_train_data = np.array(y_train_data[:2000])

x_test_data = np.array(x_test_data[:800])
y_test_data = np.array(y_test_data[:800])


# In[59]:


import math
import operator

class KNN:
    def __init__(self):
        self.k=7
    
    
    def Euclidean(self,x_test_data,x_train_data,length):
        distance=0
        for i in range(length):
            distance+=pow(x_test_data[i]-x_train_data[i],2)
        return math.sqrt(distance)
    
    def get_neighbours(self,x_train_data,x_test_data,y_train_data):
        distance=[]
        length=len(x_test_data)-1
        for i in range(len(x_train_data)):
            dist=self.Euclidean(x_test_data,x_train_data[i],length)
            distance.append((y_train_data[i],dist))
            
        distance.sort(key=operator.itemgetter(1))
        neighbour=[]
        for i in range(self.k):
            neighbour.append(distance[i][0])
        return neighbour
        
        

    def getMajority(self,neighbors):
        majority = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in majority:
                majority[response] += 1
            else:
                majority[response] = 1
        majority = sorted(majority.items(), key=operator.itemgetter(1), reverse=True)
        return majority[0][0]
  
    def getAccuracy(self,y_test_data, predictions):
        correct = 0
        for x in range(len(y_test_data)):
            if y_test_data[x] == predictions[x]:
                correct += 1
        return (correct/float(len(y_test_data))) * 100.0

        
def main(x_train_data,y_train_data,x_test_data,y_test_data):
    obj = KNN()
    
    predictions=[]
    
    for i in range(len(x_test_data)):
        neighbours = obj.get_neighbours(x_train_data,x_test_data[i],y_train_data)
#         print(neighbours)
        result = obj.getMajority(neighbours)
    
        predictions.append(result)
        print(' predicted value= ' + repr(result) + ', actual value= ' + repr(y_test_data[i]))
        
    accuracy =obj.getAccuracy(y_test_data, predictions)
    print('\nAccuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main(x_train_data,y_train_data,x_test_data,y_test_data)
    
    
    


# In[ ]:




