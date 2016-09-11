
# coding: utf-8

# ## Predicting Red Hat Business Value
# 
# 数据存放在两个文件中，一个是关于人的信息，另外一个是关于活动的信息。people file包含了人、活动时间、活动的特征信息，每个人都有一个唯一的id，每一行刻画了一个人的相关信息。activity file包含了所有的活动信息。每一行代表一个人在何时进行了怎样的活动，每个活动有一个唯一的活动id。
# 
# 任务是预测人从事特定活动后的潜在商业价值，商业价值在活动文件中定义为yes/no[0或者1]，outcome列记录了一个人从事某个活动之后是否在特定时间窗口下达成了商业价值。
# 
# 活动文件有多种类型的活动，其中Type 1和Type 2~7有所不同，因为Type 1有更多的刻画特征特征（9个）。

# In[1]:

# From：https://www.kaggle.com/abriosi/predicting-red-hat-business-value/raddar-0-98-xgboost-sparse-matrix-python
import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder


# In[2]:

# 载入数据
act_train_data = pd.read_csv("act_train.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
act_test_data  = pd.read_csv("act_test.csv", dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
people_data    = pd.read_csv("people.csv", dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])


# In[3]:

print act_train_data.head()
print act_test_data.head()
print '----------\n', people_data.head()


# In[4]:

def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset
    
def act_data_treatment(dsname):
    dataset = dsname
    
    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if dataset[col].dtype == 'object': # 将type 1解析为1
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool': # 对于布尔型数据，处理成0和1
                dataset[col] = dataset[col].astype(np.int8)
    
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    dataset = dataset.drop('date', axis = 1)
    
    return dataset


# In[5]:

act_train_data=act_train_data.drop('char_10',axis=1)
act_test_data=act_test_data.drop('char_10',axis=1)

print("Train data shape: " + format(act_train_data.shape))
print("Test data shape: " + format(act_test_data.shape))
print("People data shape: " + format(people_data.shape))

# 预处理数据
act_train_data  = act_data_treatment(act_train_data)
act_test_data   = act_data_treatment(act_test_data)
people_data = act_data_treatment(people_data)

print act_train_data.head()
print act_test_data.head()
print people_data.head()

# join到一起
train = act_train_data.merge(people_data, on='people_id', how='left', left_index=True)
test  = act_test_data.merge(people_data, on='people_id', how='left', left_index=True)

print 'after join:'
print train.head()
print 'test...\n', test.head()


# In[6]:

train=train.sort_values(['people_id'], ascending=[1])
test=test.sort_values(['people_id'], ascending=[1])

train_columns = train.columns.values
test_columns = test.columns.values
features = list(set(train_columns) & set(test_columns)) # 求交集

print train_columns, len(train_columns)
print test_columns, len(test_columns)
print features, len(features)


# In[7]:

train.fillna('NA', inplace=True)
test.fillna('NA', inplace=True)

y = train.outcome
train=train.drop('outcome',axis=1)


# In[8]:

whole = pd.concat([train,test],ignore_index=True)
print whole.head()


# In[9]:

categorical=['group_1','activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
for category in categorical:
    whole=reduce_dimen(whole,category,9999999)
    
X = whole[:len(train)]
X_test = whole[len(train):]

del train
del whole


# In[10]:

X = X.sort_values(['people_id'], ascending=[1])

X = X[features].drop(['people_id', 'activity_id'], axis = 1)
X_test = X_test[features].drop(['people_id', 'activity_id'], axis = 1)

categorical=['group_1','activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
not_categorical=[]
for category in X.columns:
    if category not in categorical:
        not_categorical.append(category)


# In[20]:

enc = OneHotEncoder(handle_unknown='ignore')
enc = enc.fit(pd.concat([X[categorical],X_test[categorical]]))
X_cat_sparse = enc.transform(X[categorical])
X_test_cat_sparse = enc.transform(X_test[categorical])
print 'X[not_categorical]'
print X[not_categorical].shape
print X_cat_sparse.shape
print 'X_cat_sparse'
print X_cat_sparse
print 'X_test_cat_sparse'
print X_test_cat_sparse


# In[14]:

from scipy.sparse import hstack
X_sparse=hstack((X[not_categorical], X_cat_sparse))
X_test_sparse=hstack((X_test[not_categorical], X_test_cat_sparse))


# In[22]:

print 'X[not_categorical]'
print X[not_categorical].shape
print X_cat_sparse.shape
print X_sparse.shape
print 'X_cat_sparse'
print X_cat_sparse
print X_sparse
print 'X_test_cat_sparse'
print X_test_cat_sparse


# In[ ]:

print("Training data: " + format(X_sparse.shape))
print("Test data: " + format(X_test_sparse.shape))
print("###########")
print("One Hot enconded Test Dataset Script")

dtrain = xgb.DMatrix(X_sparse,label=y)
dtest = xgb.DMatrix(X_test_sparse)

param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree']= 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"

watchlist  = [(dtrain,'train')]
num_round = 300
early_stopping_rounds=10
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)

ypred = bst.predict(dtest)
output = pd.DataFrame({ 'activity_id' : test['activity_id'], 'outcome': ypred })
output.head()
output.to_csv('without_leak.csv', index = False)

