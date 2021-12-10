#!/usr/bin/env python
# coding: utf-8

# Data Clean

# In[1]:


# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import multiprocessing as mp
# 79: -------------------------------------------------------------------------


# In[12]:


df_temp = pd.read_csv('train.csv').iloc[:100,:]
df_uniq = pd.read_csv('unique_m.csv').iloc[:100,:]


# In[13]:


df_temp['material'] = df_uniq['material']
df_temp = pd.DataFrame(df_temp)


# In[14]:


df_grp = df_temp.groupby('material')
materials = []
data_total = []
index = 0
for material, data in df_grp: 
    materials.append({index: material})
    
df_mtr_grp = pd.DataFrame(materials).reset_index()

train_valid, test= train_test_split(df_mtr_grp, test_size=0.1, 
                                        random_state=2021*12*1)
# split into train and valid set
kfold = KFold(n_splits = 10, shuffle = True, random_state = 2021*12*1)
index = kfold.split(X = train_valid)
train_index = []
valid_index = []
for i, j in index: 
    train_index.append(i)
    valid_index.append(j)


# In[15]:


def transfer(data, index): 
    ls_mtr = data.iloc[index, :][0].tolist()
    ls_bool = [False] * len(df_temp)
    for i in ls_mtr:
        ls_bool += df_temp['material'] == i
    x = df_temp[ls_bool].drop(columns = 'material').drop(columns = 'critical_temp')
    y = df_temp[ls_bool]['critical_temp']
    return (x, y)


# In[16]:


x_train, y_train = transfer(train_valid, train_index[0])
x_valid, y_valid = transfer(train_valid, valid_index[0])
x_test, y_test = transfer(test, range(len(test)))

print ('The lenght of train data from folder1: ',len(x_train))
print ('The lenght of valid data from folder1: ',len(x_valid))
print ('The lenght of test data: ', len(x_test))


# GBDT go fuck the great lake

# In[22]:


def gbr_job(index, q):
    # GBDT
    
    # Tune the rounds of boosting
    boost_rounds = np.arange(1,50,1)
    alpha_optim_index = []
    mse_list = []
    ls_index = [index, index + 1]
    for i in ls_index: 
        valid_mse = []
        for boost_round in boost_rounds:
            gbr = GradientBoostingRegressor(n_estimators = boost_round, 
                                            learning_rate=0.5, 
                                            random_state = 2021*12*1)
            x_train, y_train = transfer(train_valid, train_index[i])
            gbr.fit(x_train, y_train)
            x_valid, y_valid = transfer(train_valid, valid_index[i])
            y_predict = gbr.predict(x_valid)
            valid_mse.append(np.mean((y_valid - y_predict)**2))
        alpha_optim_index.append([np.argmin(valid_mse),np.min(valid_mse)])
        mse_list.append(valid_mse)
    q.put(mse_list)


# In[23]:


def multicore(): 
    q = mp.Queue()
    p1 = mp.Process(target=gbr_job, args=(0,q))
    p2 = mp.Process(target=gbr_job, args=(2,q))
    p3 = mp.Process(target=gbr_job, args=(4,q))
    p4 = mp.Process(target=gbr_job, args=(6,q))
    p5 = mp.Process(target=gbr_job, args=(8,q))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    mse_list1 = q.get()
    mse_list2 = q.get()
    mse_list3 = q.get()
    mse_list4 = q.get()
    mse_list5 = q.get()


# In[ ]:


multicore()


# In[172]:


plt.figure(figsize=(9, 5))
valid_list = [0,1,2,3,4,5,6,7,8,9]
for i in valid_list:
    plt.plot(df_mse['Boosting Rounds'],
             df_mse[i],label = 'Data_set{a}'.format(a = i))
plt.legend(loc='upper right')
plt.xlabel('Boosting Rounds')
plt.ylabel('Value of MSE')
# 79: -------------------------------------------------------------------------


# In[173]:


valid_mse = []
for i in range (10): 
    gbr = GradientBoostingRegressor(n_estimators = 20, 
                                    learning_rate=0.5, 
                                    random_state = 2021*12*1)
    x_train, y_train = transfer(train_valid, train_index[i])
    gbr.fit(x_train, y_train)
    x_valid, y_valid = transfer(train_valid, valid_index[i])
    y_predict = gbr.predict(x_valid)
    valid_mse.append(np.mean((y_valid - y_predict)**2))
model_valid_optim = np.argmin(valid_mse)
# 79: -------------------------------------------------------------------------


# In[175]:


plt.figure(figsize=(9, 12))
plt.subplot(2, 1, 1)

valid_list = list(range(1,11))
plt.bar(valid_list, valid_mse)
plt.xlabel('Data Set')
plt.ylabel('Value of MSE')
plt.title('DataSet - MSE')

print("The lowest data set with MSE is", model_valid_optim+1)
print("The MSE is",round(valid_mse[model_valid_optim],3))
# 79: -------------------------------------------------------------------------

