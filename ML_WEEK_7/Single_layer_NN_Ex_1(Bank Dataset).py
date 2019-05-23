#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


original_df = pd.read_csv("bank.csv", delimiter = ';')
dataframe = original_df


# In[3]:


dataframe.head()


# In[4]:


dataframe.describe()


# In[5]:


dataframe.shape


# In[6]:


dataframe.info()


# In[7]:


print("check for Duplicated Data",dataframe.duplicated().sum())


# In[8]:


dataframe.replace(['yes','no'],[1,0],inplace = True)
dataframe.head()


# In[9]:


dataframe.drop(['marital', 'education','poutcome','day','month'], axis=1,inplace=True)


# In[10]:


dataframe = pd.get_dummies(dataframe)
dataframe.head()


# In[11]:


dataframe.shape


# In[12]:


dataframe.columns


# In[13]:


dataframe.replace(np.NaN,dataframe.mean(),inplace= True)
dataframe.head()


# In[14]:


dataframe = ((dataframe - dataframe.min())
             /
             (dataframe.max()- dataframe.min()))
dataframe.head()


# In[15]:


dataframe.boxplot(rot=50, figsize=(10,7))


# In[16]:


def Split(data):
    train=int(0.70*len(data))
    test=int(0.30*len(data))
    return train,test

train , test = Split(dataframe)


# In[17]:


print("Train data: ",train)
print("Test data: ",test)


# In[18]:


train_data = dataframe.head(train)
test_data = dataframe.tail(test)

# print("Train_data:")
# train_data.head()

# print("test_data:")
# test_data.head()


# In[19]:


def seperate(data):
    y = data.y
    data = data.drop('y',axis = 1)
    return data,y
x_train_data,y_train_data = seperate(train_data)
x_test_data, y_test_data = seperate(test_data)

x_train_data = np.array(x_train_data)   
y_train_data = np.array(y_train_data)
x_test_data  = np.array(x_test_data)
y_test_data  = np.array(y_test_data)


# In[28]:


y_train_data = y_train_data.reshape(y_train_data.shape[0],1)
y_test_data = y_test_data.reshape(y_test_data.shape[0],1)

print("x_train_data ",x_train_data.shape)
print("y_train_data ",y_train_data.shape)
print("x_test_data ",x_test_data.shape)
print("y_test_data ",y_test_data.shape)


# In[ ]:


class Single_Layer_NN:
    
    def __init__(self):
        self.learning_rate = 0.0008
        self.epoch = 1000
        
    def gradient_descent(self,x_train_data,y_train_data):
        
        size = len(x_train_data)
#         print(size) (3164,)
        dw = 0.0
        dz = 0.0
        db = 0.0
        nshape = x_train_data.shape[1]
#         print(nshape) (3164, 22)

        w = np.full((nshape,1),0.5)
#         print(w.shape)  (22, 1)
        b = np.ones((1,1) ,dtype='float')
#         print(b)  [[1.]]
        
        for i in range(self.epoch):
            
            z = np.dot(w.T,x_train_data.T) + b
    #         print(z.shape)  (1, 3164)

            a = 1/ 1 + np.exp(-z)
    #         print(a.shape)  (1, 3164)

            dz = a - y_train_data.T
    #       print(dz.shape)  (1, 3164)

            dw = np.dot(dz,x_train_data)/size
#             print(dw.shape)  (1, 22)

            db = 1/size*np.sum(dz, axis =1 ,keepdims=True)
#         print(db.shape)  (1, 1)

            w = w - np.dot(self.learning_rate , dw.T)
            #print(w.shape)
            
            b = b - np.dot(self.learning_rate , db)
#         print(b.shape)

           
      
        return w, b
    
    def prediction(self, w, b, x_test_data):
        
        y_prediction = np.zeros((x_test_data.shape[0], 1),dtype = 'float')
        
#         print(w.shape)  (22, 1)
#         print(x_test_data.shape)  (1356, 22)

        z = np.dot(w.T, x_test_data.T) + b
#         print(z.shape)  (1, 1356)
        
        a = 1/ (1 + np.exp(-z))
#         a = pd.DataFrame(a)
#         print(a[:100])
#         print(a.shape)  (1, 1356)
#         b = a.shape[1]
        
        for i in range(0,a.shape[1]):
            if round(a[0][i],2 )<= 0.5:
                y_prediction[i][0] = 1
            else:
                y_prediction[i][0] = 0
                
        y_prediction = np.reshape(y_prediction, (len(y_prediction), 1))
#         print(y_prediction[:300])    

        return y_prediction

        
def main(dataframe,x_train_data,y_train_data,x_test_data,y_test_data):
    
    obj = Single_Layer_NN()
    
    x_train_data = np.column_stack((np.ones((x_train_data.shape[0],1)),x_train_data))
    x_test_data = np.column_stack((np.ones((x_test_data.shape[0],1)),x_test_data))
#     print(x_test_data.shape)
#     print(x_train_data.shape)
    
    w, b = obj.gradient_descent(x_train_data,y_train_data)
#     print("weight: ",w)
#     print("bias: ",b)

    y_prediction_test = obj.prediction(w, b, x_test_data)
    y_prediction_train = obj.prediction(w, b, x_train_data)
    
#     for i in range(0,300):
#         print(y_prediction[i], y_test_data[i])

    test_accuracy = (100 - np.mean(np.abs(y_prediction_test - y_test_data)) * 100)
    print("Test Accuracy : ",test_accuracy)
    
    
main(dataframe, x_train_data,y_train_data,x_test_data,y_test_data) 
        


# In[ ]:




