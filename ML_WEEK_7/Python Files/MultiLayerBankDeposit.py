#!/usr/bin/env python
# coding: utf-8

# In[361]:


import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import math
import seaborn as sb
from matplotlib import pyplot as plt


# In[362]:


# loads and read csv file
df_original=pd.read_csv("bank.csv",delimiter=";")
df =df_original
df.head()


# In[363]:


# checks data types of columns
df.dtypes


# In[364]:


# checks for null values
df.isnull().sum()


# In[365]:


df.info()


# In[366]:


df.describe()


# In[367]:


# checks for duplicate values
df.duplicated().sum()


# In[368]:


# display column names
df.columns


# In[369]:


# replacing categorical data with binary values
df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[370]:


# display shape of the dataframe
df.shape


# In[371]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr)


# In[372]:


# checks correlation with all columns
print(corr['y'].sort_values(ascending=False)[:])


# In[373]:


# return unique values in given column
df['y'].unique()


# In[374]:


df['y'].value_counts()


# In[375]:


sb.countplot(x='y',data=df,palette='hls')


# In[376]:


sb.boxplot(data=df)


# In[377]:


# display skewness of dataframe
target=df.skew()
sb.distplot(target)


# In[378]:


df.shape


# In[379]:


# convert categorical data into dummy (binary)  variables 
df=pd.get_dummies(df)


# In[380]:


# scales the limit of variable
def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df


# In[381]:


df = Feature_Scaling(df)
df.head()


# In[382]:


# splits the data in 70% & 30% format
def Split(data):
    train_set=0.70*len(data)
    train=int(train_set)
#         print(train)
    test_set=0.30*len(data)
    test=int(test_set)
        
    return train,test

train,test = Split(df)


# In[383]:


train_data=df.head(train)
test_data=df.tail(test)


# In[384]:


# Separating the output(label) and the parameters(features) of dataframe
def separate(df):
    output = df.y
    return df.drop('y', axis=1), output

x_data_train,y_data_train = separate(train_data)


# In[385]:


x_data_test,y_data_test=separate(test_data)


# In[386]:


x_data_test,y_data_test=separate(test_data)


# In[387]:


class MultiLayerNeural:
    def __init__(self):
        self.alpha = 0.221
        self.epoch = 1000
    
        
    def Train(self,x_train_data, y_train_data):
        weight=[]
        bias=[]
        
        # Initializing layers
        layers=[x_train_data.shape[1],4,5,3,1]
        db=0.0
        a = [0] * len(layers)
        z = [0] * len(layers)
        A = [0] * len(layers)
        dg = [0] * len(layers)
        da = [0] * len(layers)
        dz = [0] * len(layers)
        db = [0] * len(layers)
        dw = [0] * len(layers)
        
        a[0]=x_train_data.T
        
        # Initializing weights and bias for all layers     
        for i in range(len(layers)):
            weight.append(np.random.rand(layers[i],layers[i-1])*0.01)
            bias.append(np.zeros(((layers[i],1))))    
    
        # forward propagation         
        for length in range(self.epoch):
            for i in range(1,len(layers)):
                z[i] = np.dot(weight[i],a[i-1])+bias[i]
                a[i] = (1 / (1 + np.exp(-z[i])))
            
        # backward propagation  
            for i in reversed(range(1,len(layers))):
                da[i]=(-(y_train_data.T/a[i])+((1-y_train_data.T)/(1-a[i])))
                dg[i] = (1 / (1 + np.exp(-z[i]))) * (1 - (1 / (1 + np.exp(-z[i]))))
                dz[i]=da[i]*dg[i]
                dw[i]=(np.dot(dz[i],a[i-1].T)/len(x_train_data))
                db[i]=(np.sum(dz[i],axis=1,keepdims=True)/len(x_train_data))
                weight[i]=(weight[i]-(np.dot(self.alpha,dw[i])))
                bias[i]=(bias[i]-(np.dot(self.alpha,db[i])))
        return weight,bias

    def Test_data(self, x_test_data, weight,bias): 
        layers=[x_test_data.shape[1],4,5,3,1]
        
        a = [0] * len(layers)
        z = [0] * len(layers)
        a[0]=x_test_data.T
        
        # testing test data on Activation/Hypothesis Function 
        for i in range(1,len(layers)):
            z[i] = np.dot(weight[i],a[i-1])+bias[i]
            a[i] = (1 / (1 + np.exp(-z[i])))
#         print(len(a))
        return a[-1]
           
    def Accuracy(self, y_test_data, y_predict):
        y_predict = np.nan_to_num(y_predict)
   
        test_accuracy = 100 - (np.mean(np.abs(y_predict - y_test_data)) * 100)        
        return test_accuracy

def main():
    # creates class object 
    obj = MultiLayerNeural()
    
    # convert data into numpy array     
    x_train_data = np.array(x_data_train)
    y_train_data = np.array(y_data_train)
    y_train_data = y_train_data.reshape(len(y_train_data),1)
        
    x_test_data = np.array(x_data_test)
    y_test_data = np.array(y_data_test)
    y_test_data = y_test_data.reshape(len(y_test_data),1)
    
#     print("x_train_data",x_train_data.shape)
#     print("y_train_data",y_train_data.shape) 
#     print("x_test_data",x_test_data.shape)
#     print("y_test_data",y_test_data.shape)  
    
       
    # calling method by class object to get weights and bias
    weights,b = obj.Train(x_train_data, y_train_data)

    # getting prediction values    
    y_predict_train = obj.Test_data(x_train_data, weights,b)
    y_predict_test = obj.Test_data(x_test_data, weights,b)

    # getting accuracy     
    acc_train=obj.Accuracy(y_predict_train,y_train_data)
    print("\n\naccuracy of train data=",acc_train)
    
    acc_test=obj.Accuracy(y_predict_test,y_test_data)
    print("accuracy of test data=",acc_test)
    
if __name__ == '__main__':
    main()
    

