
## Predicting Red Hat Business Value

数据存放在两个文件中，一个是关于人的信息，另外一个是关于活动的信息。people file包含了人、活动时间、活动的特征信息，每个人都有一个唯一的id，每一行刻画了一个人的相关信息。activity file包含了所有的活动信息。每一行代表一个人在何时进行了怎样的活动，每个活动有一个唯一的活动id。

任务是预测人从事特定活动后的潜在商业价值，商业价值在活动文件中定义为yes/no[0或者1]，outcome列记录了一个人从事某个活动之后是否在特定时间窗口下达成了商业价值。

活动文件有多种类型的活动，其中Type 1和Type 2~7有所不同，因为Type 1有更多的刻画特征特征（9个）。


```python
# From：https://www.kaggle.com/abriosi/predicting-red-hat-business-value/raddar-0-98-xgboost-sparse-matrix-python
import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
```


```python
# 载入数据
act_train_data = pd.read_csv("act_train.csv",dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
act_test_data  = pd.read_csv("act_test.csv", dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
people_data    = pd.read_csv("people.csv", dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])
```


```python
print act_train_data.head()
print act_test_data.head()
print '----------\n', people_data.head()
```

      people_id   activity_id       date activity_category char_1 char_2 char_3  \
    0   ppl_100  act2_1734928 2023-08-26            type 4    NaN    NaN    NaN   
    1   ppl_100  act2_2434093 2022-09-27            type 2    NaN    NaN    NaN   
    2   ppl_100  act2_3404049 2022-09-27            type 2    NaN    NaN    NaN   
    3   ppl_100  act2_3651215 2023-08-04            type 2    NaN    NaN    NaN   
    4   ppl_100  act2_4109017 2023-08-26            type 2    NaN    NaN    NaN   
    
      char_4 char_5 char_6 char_7 char_8 char_9  char_10  outcome  
    0    NaN    NaN    NaN    NaN    NaN    NaN  type 76        0  
    1    NaN    NaN    NaN    NaN    NaN    NaN   type 1        0  
    2    NaN    NaN    NaN    NaN    NaN    NaN   type 1        0  
    3    NaN    NaN    NaN    NaN    NaN    NaN   type 1        0  
    4    NaN    NaN    NaN    NaN    NaN    NaN   type 1        0  
        people_id   activity_id       date activity_category   char_1   char_2  \
    0  ppl_100004   act1_249281 2022-07-20            type 1   type 5  type 10   
    1  ppl_100004   act2_230855 2022-07-20            type 5      NaN      NaN   
    2   ppl_10001   act1_240724 2022-10-14            type 1  type 12   type 1   
    3   ppl_10001    act1_83552 2022-11-27            type 1  type 20  type 10   
    4   ppl_10001  act2_1043301 2022-10-15            type 5      NaN      NaN   
    
       char_3  char_4  char_5  char_6  char_7   char_8   char_9    char_10  
    0  type 5  type 1  type 6  type 1  type 1   type 7   type 4        NaN  
    1     NaN     NaN     NaN     NaN     NaN      NaN      NaN   type 682  
    2  type 5  type 4  type 6  type 1  type 1  type 13  type 10        NaN  
    3  type 5  type 4  type 6  type 1  type 1   type 5   type 5        NaN  
    4     NaN     NaN     NaN     NaN     NaN      NaN      NaN  type 3015  
    ----------
        people_id  char_1      group_1  char_2       date   char_3   char_4  \
    0     ppl_100  type 2  group 17304  type 2 2021-06-29   type 5   type 5   
    1  ppl_100002  type 2   group 8688  type 3 2021-01-06  type 28   type 9   
    2  ppl_100003  type 2  group 33592  type 3 2022-06-10   type 4   type 8   
    3  ppl_100004  type 2  group 22593  type 3 2022-07-20  type 40  type 25   
    4  ppl_100006  type 2   group 6534  type 3 2022-07-27  type 40  type 25   
    
       char_5  char_6   char_7   ...   char_29 char_30 char_31 char_32 char_33  \
    0  type 5  type 3  type 11   ...     False    True    True   False   False   
    1  type 5  type 3  type 11   ...     False    True    True    True    True   
    2  type 5  type 2   type 5   ...     False   False    True    True    True   
    3  type 9  type 4  type 16   ...      True    True    True    True    True   
    4  type 9  type 3   type 8   ...     False   False    True   False   False   
    
      char_34 char_35 char_36 char_37 char_38  
    0    True    True    True   False      36  
    1    True    True    True   False      76  
    2    True   False    True    True      99  
    3    True    True    True    True      76  
    4   False    True    True   False      84  
    
    [5 rows x 41 columns]



```python
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
```


```python
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
```

    Train data shape: (2197291, 14)
    Test data shape: (498687, 13)
    People data shape: (189118, 41)
      people_id   activity_id  activity_category  char_1  char_2  char_3  char_4  \
    0   ppl_100  act2_1734928                  4       0       0       0       0   
    1   ppl_100  act2_2434093                  2       0       0       0       0   
    2   ppl_100  act2_3404049                  2       0       0       0       0   
    3   ppl_100  act2_3651215                  2       0       0       0       0   
    4   ppl_100  act2_4109017                  2       0       0       0       0   
    
       char_5  char_6  char_7  char_8  char_9  outcome  year  month  day  \
    0       0       0       0       0       0        0  2023      8   26   
    1       0       0       0       0       0        0  2022      9   27   
    2       0       0       0       0       0        0  2022      9   27   
    3       0       0       0       0       0        0  2023      8    4   
    4       0       0       0       0       0        0  2023      8   26   
    
       isweekend  
    0          1  
    1          0  
    2          0  
    3          0  
    4          1  
        people_id   activity_id  activity_category  char_1  char_2  char_3  \
    0  ppl_100004   act1_249281                  1       5      10       5   
    1  ppl_100004   act2_230855                  5       0       0       0   
    2   ppl_10001   act1_240724                  1      12       1       5   
    3   ppl_10001    act1_83552                  1      20      10       5   
    4   ppl_10001  act2_1043301                  5       0       0       0   
    
       char_4  char_5  char_6  char_7  char_8  char_9  year  month  day  isweekend  
    0       1       6       1       1       7       4  2022      7   20          0  
    1       0       0       0       0       0       0  2022      7   20          0  
    2       4       6       1       1      13      10  2022     10   14          0  
    3       4       6       1       1       5       5  2022     11   27          1  
    4       0       0       0       0       0       0  2022     10   15          1  
        people_id  char_1  group_1  char_2  char_3  char_4  char_5  char_6  \
    0     ppl_100       2    17304       2       5       5       5       3   
    1  ppl_100002       2     8688       3      28       9       5       3   
    2  ppl_100003       2    33592       3       4       8       5       2   
    3  ppl_100004       2    22593       3      40      25       9       4   
    4  ppl_100006       2     6534       3      40      25       9       3   
    
       char_7  char_8    ...      char_33  char_34  char_35  char_36  char_37  \
    0      11       2    ...            0        1        1        1        0   
    1      11       2    ...            1        1        1        1        0   
    2       5       2    ...            1        1        0        1        1   
    3      16       2    ...            1        1        1        1        1   
    4       8       2    ...            0        0        1        1        0   
    
       char_38  year  month  day  isweekend  
    0       36  2021      6   29          0  
    1       76  2021      1    6          0  
    2       99  2022      6   10          0  
    3       76  2022      7   20          0  
    4       84  2022      7   27          0  
    
    [5 rows x 44 columns]
    after join:
      people_id   activity_id  activity_category  char_1_x  char_2_x  char_3_x  \
    0   ppl_100  act2_1734928                  4         0         0         0   
    0   ppl_100  act2_2434093                  2         0         0         0   
    0   ppl_100  act2_3404049                  2         0         0         0   
    0   ppl_100  act2_3651215                  2         0         0         0   
    0   ppl_100  act2_4109017                  2         0         0         0   
    
       char_4_x  char_5_x  char_6_x  char_7_x     ...       char_33  char_34  \
    0         0         0         0         0     ...             0        1   
    0         0         0         0         0     ...             0        1   
    0         0         0         0         0     ...             0        1   
    0         0         0         0         0     ...             0        1   
    0         0         0         0         0     ...             0        1   
    
       char_35  char_36  char_37  char_38  year_y  month_y  day_y  isweekend_y  
    0        1        1        0       36    2021        6     29            0  
    0        1        1        0       36    2021        6     29            0  
    0        1        1        0       36    2021        6     29            0  
    0        1        1        0       36    2021        6     29            0  
    0        1        1        0       36    2021        6     29            0  
    
    [5 rows x 60 columns]
    test...
        people_id   activity_id  activity_category  char_1_x  char_2_x  char_3_x  \
    3  ppl_100004   act1_249281                  1         5        10         5   
    3  ppl_100004   act2_230855                  5         0         0         0   
    5   ppl_10001   act1_240724                  1        12         1         5   
    5   ppl_10001    act1_83552                  1        20        10         5   
    5   ppl_10001  act2_1043301                  5         0         0         0   
    
       char_4_x  char_5_x  char_6_x  char_7_x     ...       char_33  char_34  \
    3         1         6         1         1     ...             1        1   
    3         0         0         0         0     ...             1        1   
    5         4         6         1         1     ...             1        1   
    5         4         6         1         1     ...             1        1   
    5         0         0         0         0     ...             1        1   
    
       char_35  char_36  char_37  char_38  year_y  month_y  day_y  isweekend_y  
    3        1        1        1       76    2022        7     20            0  
    3        1        1        1       76    2022        7     20            0  
    5        1        1        1       90    2022       10     14            0  
    5        1        1        1       90    2022       10     14            0  
    5        1        1        1       90    2022       10     14            0  
    
    [5 rows x 59 columns]



```python
train=train.sort_values(['people_id'], ascending=[1])
test=test.sort_values(['people_id'], ascending=[1])

train_columns = train.columns.values
test_columns = test.columns.values
features = list(set(train_columns) & set(test_columns)) # 求交集

print train_columns, len(train_columns)
print test_columns, len(test_columns)
print features, len(features)
```

    ['people_id' 'activity_id' 'activity_category' 'char_1_x' 'char_2_x'
     'char_3_x' 'char_4_x' 'char_5_x' 'char_6_x' 'char_7_x' 'char_8_x'
     'char_9_x' 'outcome' 'year_x' 'month_x' 'day_x' 'isweekend_x' 'char_1_y'
     'group_1' 'char_2_y' 'char_3_y' 'char_4_y' 'char_5_y' 'char_6_y'
     'char_7_y' 'char_8_y' 'char_9_y' 'char_10' 'char_11' 'char_12' 'char_13'
     'char_14' 'char_15' 'char_16' 'char_17' 'char_18' 'char_19' 'char_20'
     'char_21' 'char_22' 'char_23' 'char_24' 'char_25' 'char_26' 'char_27'
     'char_28' 'char_29' 'char_30' 'char_31' 'char_32' 'char_33' 'char_34'
     'char_35' 'char_36' 'char_37' 'char_38' 'year_y' 'month_y' 'day_y'
     'isweekend_y'] 60
    ['people_id' 'activity_id' 'activity_category' 'char_1_x' 'char_2_x'
     'char_3_x' 'char_4_x' 'char_5_x' 'char_6_x' 'char_7_x' 'char_8_x'
     'char_9_x' 'year_x' 'month_x' 'day_x' 'isweekend_x' 'char_1_y' 'group_1'
     'char_2_y' 'char_3_y' 'char_4_y' 'char_5_y' 'char_6_y' 'char_7_y'
     'char_8_y' 'char_9_y' 'char_10' 'char_11' 'char_12' 'char_13' 'char_14'
     'char_15' 'char_16' 'char_17' 'char_18' 'char_19' 'char_20' 'char_21'
     'char_22' 'char_23' 'char_24' 'char_25' 'char_26' 'char_27' 'char_28'
     'char_29' 'char_30' 'char_31' 'char_32' 'char_33' 'char_34' 'char_35'
     'char_36' 'char_37' 'char_38' 'year_y' 'month_y' 'day_y' 'isweekend_y'] 59
    ['year_x', 'char_3_y', 'char_10', 'people_id', 'char_33', 'char_2_y', 'group_1', 'activity_category', 'char_38', 'char_1_y', 'char_8_y', 'char_19', 'char_18', 'char_17', 'char_8_x', 'char_15', 'char_14', 'char_9_y', 'char_9_x', 'char_11', 'char_1_x', 'char_26', 'activity_id', 'char_3_x', 'isweekend_x', 'char_6_y', 'year_y', 'char_5_y', 'char_31', 'char_25', 'char_12', 'char_30', 'char_2_x', 'char_35', 'char_16', 'isweekend_y', 'char_27', 'char_24', 'char_32', 'char_22', 'char_23', 'char_20', 'char_5_x', 'char_4_x', 'char_13', 'char_21', 'char_28', 'char_29', 'day_x', 'day_y', 'char_37', 'char_6_x', 'char_4_y', 'char_34', 'char_36', 'month_x', 'month_y', 'char_7_y', 'char_7_x'] 59



```python
train.fillna('NA', inplace=True)
test.fillna('NA', inplace=True)

y = train.outcome
train=train.drop('outcome',axis=1)
```


```python
whole = pd.concat([train,test],ignore_index=True)
print whole.head()
```

      people_id   activity_id  activity_category  char_1_x  char_2_x  char_3_x  \
    0   ppl_100  act2_1734928                  4         0         0         0   
    1   ppl_100  act2_2434093                  2         0         0         0   
    2   ppl_100  act2_3404049                  2         0         0         0   
    3   ppl_100  act2_3651215                  2         0         0         0   
    4   ppl_100  act2_4109017                  2         0         0         0   
    
       char_4_x  char_5_x  char_6_x  char_7_x     ...       char_33  char_34  \
    0         0         0         0         0     ...             0        1   
    1         0         0         0         0     ...             0        1   
    2         0         0         0         0     ...             0        1   
    3         0         0         0         0     ...             0        1   
    4         0         0         0         0     ...             0        1   
    
       char_35  char_36  char_37  char_38  year_y  month_y  day_y  isweekend_y  
    0        1        1        0       36    2021        6     29            0  
    1        1        1        0       36    2021        6     29            0  
    2        1        1        0       36    2021        6     29            0  
    3        1        1        0       36    2021        6     29            0  
    4        1        1        0       36    2021        6     29            0  
    
    [5 rows x 59 columns]



```python
categorical=['group_1','activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
for category in categorical:
    whole=reduce_dimen(whole,category,9999999)
    
X = whole[:len(train)]
X_test = whole[len(train):]

del train
del whole
```


```python
X = X.sort_values(['people_id'], ascending=[1])

X = X[features].drop(['people_id', 'activity_id'], axis = 1)
X_test = X_test[features].drop(['people_id', 'activity_id'], axis = 1)

categorical=['group_1','activity_category','char_1_x','char_2_x','char_3_x','char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x','char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y','char_8_y','char_9_y']
not_categorical=[]
for category in X.columns:
    if category not in categorical:
        not_categorical.append(category)
```


```python
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
```

    X[not_categorical]
    (2197291, 38)
    (2197291, 31233)
    X_cat_sparse
      (0, 31225)	1.0
      (0, 31217)	1.0
      (0, 31201)	1.0
      (0, 31186)	1.0
      (0, 31179)	1.0
      (0, 31154)	1.0
      (0, 31111)	1.0
      (0, 31105)	1.0
      (0, 31084)	1.0
      (0, 31065)	1.0
      (0, 31056)	1.0
      (0, 31050)	1.0
      (0, 31042)	1.0
      (0, 31034)	1.0
      (0, 31022)	1.0
      (0, 30989)	1.0
      (0, 30937)	1.0
      (0, 30933)	1.0
      (0, 10990)	1.0
      (1, 31225)	1.0
      (1, 31217)	1.0
      (1, 31201)	1.0
      (1, 31186)	1.0
      (1, 31179)	1.0
      (1, 31154)	1.0
      :	:
      (2197289, 31034)	1.0
      (2197289, 31022)	1.0
      (2197289, 30989)	1.0
      (2197289, 30937)	1.0
      (2197289, 30931)	1.0
      (2197289, 11348)	1.0
      (2197290, 31225)	1.0
      (2197290, 31217)	1.0
      (2197290, 31192)	1.0
      (2197290, 31184)	1.0
      (2197290, 31176)	1.0
      (2197290, 31156)	1.0
      (2197290, 31108)	1.0
      (2197290, 31106)	1.0
      (2197290, 31084)	1.0
      (2197290, 31065)	1.0
      (2197290, 31056)	1.0
      (2197290, 31050)	1.0
      (2197290, 31042)	1.0
      (2197290, 31034)	1.0
      (2197290, 31022)	1.0
      (2197290, 30989)	1.0
      (2197290, 30937)	1.0
      (2197290, 30931)	1.0
      (2197290, 11348)	1.0
    X_test_cat_sparse
      (0, 31225)	1.0
      (0, 31217)	1.0
      (0, 31206)	1.0
      (0, 31187)	1.0
      (0, 31183)	1.0
      (0, 31174)	1.0
      (0, 31146)	1.0
      (0, 31106)	1.0
      (0, 31088)	1.0
      (0, 31072)	1.0
      (0, 31057)	1.0
      (0, 31051)	1.0
      (0, 31048)	1.0
      (0, 31035)	1.0
      (0, 31027)	1.0
      (0, 30999)	1.0
      (0, 30942)	1.0
      (0, 30930)	1.0
      (0, 14834)	1.0
      (1, 31225)	1.0
      (1, 31217)	1.0
      (1, 31206)	1.0
      (1, 31187)	1.0
      (1, 31183)	1.0
      (1, 31174)	1.0
      :	:
      (498685, 31034)	1.0
      (498685, 31022)	1.0
      (498685, 30989)	1.0
      (498685, 30937)	1.0
      (498685, 30931)	1.0
      (498685, 10990)	1.0
      (498686, 31229)	1.0
      (498686, 31221)	1.0
      (498686, 31198)	1.0
      (498686, 31186)	1.0
      (498686, 31183)	1.0
      (498686, 31174)	1.0
      (498686, 31146)	1.0
      (498686, 31105)	1.0
      (498686, 31084)	1.0
      (498686, 31065)	1.0
      (498686, 31056)	1.0
      (498686, 31050)	1.0
      (498686, 31042)	1.0
      (498686, 31034)	1.0
      (498686, 31022)	1.0
      (498686, 30989)	1.0
      (498686, 30937)	1.0
      (498686, 30931)	1.0
      (498686, 10990)	1.0



```python
from scipy.sparse import hstack
X_sparse=hstack((X[not_categorical], X_cat_sparse))
X_test_sparse=hstack((X_test[not_categorical], X_test_cat_sparse))
```


```python
print 'X[not_categorical]'
print X[not_categorical].shape
print X_cat_sparse.shape
print X_sparse.shape
print 'X_cat_sparse'
print X_cat_sparse
print X_sparse
print 'X_test_cat_sparse'
print X_test_cat_sparse
```

    X[not_categorical]
    (2197291, 38)
    (2197291, 31233)
    (2197291, 31271)
    X_cat_sparse
      (0, 31225)	1.0
      (0, 31217)	1.0
      (0, 31201)	1.0
      (0, 31186)	1.0
      (0, 31179)	1.0
      (0, 31154)	1.0
      (0, 31111)	1.0
      (0, 31105)	1.0
      (0, 31084)	1.0
      (0, 31065)	1.0
      (0, 31056)	1.0
      (0, 31050)	1.0
      (0, 31042)	1.0
      (0, 31034)	1.0
      (0, 31022)	1.0
      (0, 30989)	1.0
      (0, 30937)	1.0
      (0, 30933)	1.0
      (0, 10990)	1.0
      (1, 31225)	1.0
      (1, 31217)	1.0
      (1, 31201)	1.0
      (1, 31186)	1.0
      (1, 31179)	1.0
      (1, 31154)	1.0
      :	:
      (2197289, 31034)	1.0
      (2197289, 31022)	1.0
      (2197289, 30989)	1.0
      (2197289, 30937)	1.0
      (2197289, 30931)	1.0
      (2197289, 11348)	1.0
      (2197290, 31225)	1.0
      (2197290, 31217)	1.0
      (2197290, 31192)	1.0
      (2197290, 31184)	1.0
      (2197290, 31176)	1.0
      (2197290, 31156)	1.0
      (2197290, 31108)	1.0
      (2197290, 31106)	1.0
      (2197290, 31084)	1.0
      (2197290, 31065)	1.0
      (2197290, 31056)	1.0
      (2197290, 31050)	1.0
      (2197290, 31042)	1.0
      (2197290, 31034)	1.0
      (2197290, 31022)	1.0
      (2197290, 30989)	1.0
      (2197290, 30937)	1.0
      (2197290, 30931)	1.0
      (2197290, 11348)	1.0
      (0, 0)	2023.0
      (0, 1)	1.0
      (0, 3)	36.0
      (0, 4)	2.0
      (0, 9)	1.0
      (0, 12)	1.0
      (0, 13)	2021.0
      (0, 14)	1.0
      (0, 17)	1.0
      (0, 18)	1.0
      (0, 19)	1.0
      (0, 21)	1.0
      (0, 27)	1.0
      (0, 28)	1.0
      (0, 29)	1.0
      (0, 31)	26.0
      (0, 32)	29.0
      (0, 34)	1.0
      (0, 35)	1.0
      (0, 36)	8.0
      (0, 37)	6.0
      (1, 0)	2022.0
      (1, 1)	1.0
      (1, 3)	36.0
      (1, 4)	2.0
      :	:
      (2197289, 31072)	1.0
      (2197289, 31060)	1.0
      (2197289, 31027)	1.0
      (2197289, 30975)	1.0
      (2197289, 30969)	1.0
      (2197289, 11386)	1.0
      (2197290, 31263)	1.0
      (2197290, 31255)	1.0
      (2197290, 31230)	1.0
      (2197290, 31222)	1.0
      (2197290, 31214)	1.0
      (2197290, 31194)	1.0
      (2197290, 31146)	1.0
      (2197290, 31144)	1.0
      (2197290, 31122)	1.0
      (2197290, 31103)	1.0
      (2197290, 31094)	1.0
      (2197290, 31088)	1.0
      (2197290, 31080)	1.0
      (2197290, 31072)	1.0
      (2197290, 31060)	1.0
      (2197290, 31027)	1.0
      (2197290, 30975)	1.0
      (2197290, 30969)	1.0
      (2197290, 11386)	1.0
    X_test_cat_sparse
      (0, 31225)	1.0
      (0, 31217)	1.0
      (0, 31206)	1.0
      (0, 31187)	1.0
      (0, 31183)	1.0
      (0, 31174)	1.0
      (0, 31146)	1.0
      (0, 31106)	1.0
      (0, 31088)	1.0
      (0, 31072)	1.0
      (0, 31057)	1.0
      (0, 31051)	1.0
      (0, 31048)	1.0
      (0, 31035)	1.0
      (0, 31027)	1.0
      (0, 30999)	1.0
      (0, 30942)	1.0
      (0, 30930)	1.0
      (0, 14834)	1.0
      (1, 31225)	1.0
      (1, 31217)	1.0
      (1, 31206)	1.0
      (1, 31187)	1.0
      (1, 31183)	1.0
      (1, 31174)	1.0
      :	:
      (498685, 31034)	1.0
      (498685, 31022)	1.0
      (498685, 30989)	1.0
      (498685, 30937)	1.0
      (498685, 30931)	1.0
      (498685, 10990)	1.0
      (498686, 31229)	1.0
      (498686, 31221)	1.0
      (498686, 31198)	1.0
      (498686, 31186)	1.0
      (498686, 31183)	1.0
      (498686, 31174)	1.0
      (498686, 31146)	1.0
      (498686, 31105)	1.0
      (498686, 31084)	1.0
      (498686, 31065)	1.0
      (498686, 31056)	1.0
      (498686, 31050)	1.0
      (498686, 31042)	1.0
      (498686, 31034)	1.0
      (498686, 31022)	1.0
      (498686, 30989)	1.0
      (498686, 30937)	1.0
      (498686, 30931)	1.0
      (498686, 10990)	1.0



```python
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
```
