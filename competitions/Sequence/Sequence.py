
# coding: utf-8

# ## 数字序列预测
# 
# [Kaggle地址](https://www.kaggle.com/c/integer-sequence-learning)

# In[1]:


get_ipython().magic(u'matplotlib inline')

import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[29]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[31]:

last = test.Sequence.apply(lambda x: pd.Series(x.split(','))).mode(axis=1).fillna(0)


# In[17]:

submission = pd.DataFrame({'Id': test['Id'], 'Last': last[0]})
submission.to_csv('mode.csv', index=False)

