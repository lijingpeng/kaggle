
# coding: utf-8

# ## 数据概览

# In[15]:

import pandas as pd
import numpy as np

# 载入数据
train = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/San_Francisco_Crime_Classification/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/San_Francisco_Crime_Classification/test.csv', parse_dates = ['Dates'])


# 预览训练集

# In[16]:

print train.head(10)


# 预览测试集合

# In[17]:

print test.head(10)


# 我们看到训练集和测试集都有Dates、DayOfWeek、PdDistrict三个特征，我们先从这三个特征入手。训练集中的Category是我们的预测目标，我们先对其进行编码，这里用到sklearn的LabelEncoder()，示例如下：

# In[18]:

from sklearn import preprocessing
label = preprocessing.LabelEncoder()
label.fit([1, 2, 2, 6])
print label.transform([1, 1, 2, 6]) 


# 接下来我们对类别进行编码：

# In[19]:

crime = label.fit_transform(train.Category)


# 对于离散化的特征，有一种常用的特征处理方式是二值化处理，pandas中有get_dummies()函数，函数示例如下：

# In[20]:

pd.get_dummies(pd.Series(list('abca')))


# 接下来对Dates、DayOfWeek、PdDistrict三个特征进行二值化处理：

# In[21]:

days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour) 


# 接下来重新组合训练集，并把类别附加上：

# In[22]:

train_data = pd.concat([days, district, hour], axis=1)
train_data['crime'] = crime


# 针对测试集做同样的处理：

# In[23]:

days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Dates.dt.hour) 
test_data = pd.concat([days, district, hour], axis=1)


# 预览新的训练集和测试集：

# In[24]:

print train_data.head(10)
print test_data.head(10)


# 分割训练集和验证集(70%训练,30%验证)准备建模：

# In[35]:

from sklearn.cross_validation import train_test_split
training, validation = train_test_split(train_data, train_size=0.6)


# ## 贝叶斯训练

# In[37]:

from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
feature_list = training.columns.tolist()
feature_list = feature_list[:len(feature_list) - 1]
print '选取的特征列：', feature_list
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print "朴素贝叶斯log损失为 %f" % (log_loss(validation['crime'], predicted))


# ## 逻辑回归

# In[38]:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1)
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print "逻辑回归log损失为 %f" %(log_loss(validation['crime'], predicted))


# 在测试集上运行：

# In[41]:

test_predicted = np.array(model.predict_proba(test_data[feature_list]))


# 保存结果：

# In[44]:

col_names = np.sort(train['Category'].unique())
print col_names
result = pd.DataFrame(data=test_predicted, columns=col_names)
result['Id'] = test['Id'].astype(int)
result.to_csv('output.csv', index=False)

