[kaggle地址](https://www.kaggle.com/c/sf-crime)  
[github地址](https://github.com/lijingpeng/kaggle)

特点：  
1. 离散特征
2. 离散特征二值化处理

## 数据概览


```python
import pandas as pd
import numpy as np

# 载入数据
train = pd.read_csv('~/kaggle/dataset/San_Francisco_Crime_Classification/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('~/kaggle/dataset/San_Francisco_Crime_Classification/test.csv', parse_dates = ['Dates'])
```

预览训练集


```python
print train.head(10)
```

                    Dates        Category                        Descript  \
    0 2015-05-13 23:53:00        WARRANTS                  WARRANT ARREST   
    1 2015-05-13 23:53:00  OTHER OFFENSES        TRAFFIC VIOLATION ARREST   
    2 2015-05-13 23:33:00  OTHER OFFENSES        TRAFFIC VIOLATION ARREST   
    3 2015-05-13 23:30:00   LARCENY/THEFT    GRAND THEFT FROM LOCKED AUTO   
    4 2015-05-13 23:30:00   LARCENY/THEFT    GRAND THEFT FROM LOCKED AUTO   
    5 2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM UNLOCKED AUTO   
    6 2015-05-13 23:30:00   VEHICLE THEFT               STOLEN AUTOMOBILE   
    7 2015-05-13 23:30:00   VEHICLE THEFT               STOLEN AUTOMOBILE   
    8 2015-05-13 23:00:00   LARCENY/THEFT    GRAND THEFT FROM LOCKED AUTO   
    9 2015-05-13 23:00:00   LARCENY/THEFT    GRAND THEFT FROM LOCKED AUTO   

       DayOfWeek PdDistrict      Resolution                        Address  \
    0  Wednesday   NORTHERN  ARREST, BOOKED             OAK ST / LAGUNA ST   
    1  Wednesday   NORTHERN  ARREST, BOOKED             OAK ST / LAGUNA ST   
    2  Wednesday   NORTHERN  ARREST, BOOKED      VANNESS AV / GREENWICH ST   
    3  Wednesday   NORTHERN            NONE       1500 Block of LOMBARD ST   
    4  Wednesday       PARK            NONE      100 Block of BRODERICK ST   
    5  Wednesday  INGLESIDE            NONE            0 Block of TEDDY AV   
    6  Wednesday  INGLESIDE            NONE            AVALON AV / PERU AV   
    7  Wednesday    BAYVIEW            NONE       KIRKWOOD AV / DONAHUE ST   
    8  Wednesday   RICHMOND            NONE           600 Block of 47TH AV   
    9  Wednesday    CENTRAL            NONE  JEFFERSON ST / LEAVENWORTH ST   

                X          Y  
    0 -122.425892  37.774599  
    1 -122.425892  37.774599  
    2 -122.424363  37.800414  
    3 -122.426995  37.800873  
    4 -122.438738  37.771541  
    5 -122.403252  37.713431  
    6 -122.423327  37.725138  
    7 -122.371274  37.727564  
    8 -122.508194  37.776601  
    9 -122.419088  37.807802  


预览测试集合


```python
print test.head(10)
```

       Id               Dates DayOfWeek PdDistrict                   Address  \
    0   0 2015-05-10 23:59:00    Sunday    BAYVIEW   2000 Block of THOMAS AV   
    1   1 2015-05-10 23:51:00    Sunday    BAYVIEW        3RD ST / REVERE AV   
    2   2 2015-05-10 23:50:00    Sunday   NORTHERN    2000 Block of GOUGH ST   
    3   3 2015-05-10 23:45:00    Sunday  INGLESIDE  4700 Block of MISSION ST   
    4   4 2015-05-10 23:45:00    Sunday  INGLESIDE  4700 Block of MISSION ST   
    5   5 2015-05-10 23:40:00    Sunday    TARAVAL     BROAD ST / CAPITOL AV   
    6   6 2015-05-10 23:30:00    Sunday  INGLESIDE   100 Block of CHENERY ST   
    7   7 2015-05-10 23:30:00    Sunday  INGLESIDE     200 Block of BANKS ST   
    8   8 2015-05-10 23:10:00    Sunday    MISSION     2900 Block of 16TH ST   
    9   9 2015-05-10 23:10:00    Sunday    CENTRAL      TAYLOR ST / GREEN ST   

                X          Y  
    0 -122.399588  37.735051  
    1 -122.391523  37.732432  
    2 -122.426002  37.792212  
    3 -122.437394  37.721412  
    4 -122.437394  37.721412  
    5 -122.459024  37.713172  
    6 -122.425616  37.739351  
    7 -122.412652  37.739750  
    8 -122.418700  37.765165  
    9 -122.413935  37.798886  


我们看到训练集和测试集都有Dates、DayOfWeek、PdDistrict三个特征，我们先从这三个特征入手。训练集中的Category是我们的预测目标，我们先对其进行编码，这里用到sklearn的LabelEncoder()，示例如下：


```python
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
label.fit([1, 2, 2, 6])
print label.transform([1, 1, 2, 6])
```

    [0 0 1 2]


接下来我们对类别进行编码：


```python
crime = label.fit_transform(train.Category)
```

对于离散化的特征，有一种常用的特征处理方式是二值化处理，pandas中有get_dummies()函数，函数示例如下：


```python
pd.get_dummies(pd.Series(list('abca')))
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



接下来对Dates、DayOfWeek、PdDistrict三个特征进行二值化处理：


```python
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour)
```

接下来重新组合训练集，并把类别附加上：


```python
train_data = pd.concat([days, district, hour], axis=1)
train_data['crime'] = crime
```

针对测试集做同样的处理：


```python
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Dates.dt.hour)
test_data = pd.concat([days, district, hour], axis=1)
```

预览新的训练集和测试集：


```python
print train_data.head(10)
print test_data.head(10)
```

       Friday  Monday  Saturday  Sunday  Thursday  Tuesday  Wednesday  BAYVIEW  \
    0     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    1     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    2     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    3     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    4     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    5     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    6     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    7     0.0     0.0       0.0     0.0       0.0      0.0        1.0      1.0   
    8     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   
    9     0.0     0.0       0.0     0.0       0.0      0.0        1.0      0.0   

       CENTRAL  INGLESIDE  ...     15   16   17   18   19   20   21   22   23  \
    0      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    1      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    2      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    3      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    4      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    5      0.0        1.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    6      0.0        1.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    7      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    8      0.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    9      1.0        0.0  ...    0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   

       crime  
    0     37  
    1     21  
    2     21  
    3     16  
    4     16  
    5     16  
    6     36  
    7     36  
    8     16  
    9     16  

    [10 rows x 42 columns]
       Friday  Monday  Saturday  Sunday  Thursday  Tuesday  Wednesday  BAYVIEW  \
    0     0.0     0.0       0.0     1.0       0.0      0.0        0.0      1.0   
    1     0.0     0.0       0.0     1.0       0.0      0.0        0.0      1.0   
    2     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    3     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    4     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    5     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    6     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    7     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    8     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   
    9     0.0     0.0       0.0     1.0       0.0      0.0        0.0      0.0   

       CENTRAL  INGLESIDE ...    14   15   16   17   18   19   20   21   22   23  
    0      0.0        0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    1      0.0        0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    2      0.0        0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    3      0.0        1.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    4      0.0        1.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    5      0.0        0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    6      0.0        1.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    7      0.0        1.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    8      0.0        0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  
    9      1.0        0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  

    [10 rows x 41 columns]


分割训练集和验证集(70%训练,30%验证)准备建模：


```python
from sklearn.cross_validation import train_test_split
training, validation = train_test_split(train_data, train_size=0.6)
```

## 贝叶斯训练


```python
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
feature_list = training.columns.tolist()
feature_list = feature_list[:len(feature_list) - 1]
print '选取的特征列：', feature_list
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print "朴素贝叶斯log损失为 %f" % (log_loss(validation['crime'], predicted))
```

    选取的特征列： ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    朴素贝叶斯log损失为 2.581561


## 逻辑回归


```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1)
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print "逻辑回归log损失为 %f" %(log_loss(validation['crime'], predicted))
```

    逻辑回归log损失为 2.580102


在测试集上运行：


```python
test_predicted = np.array(model.predict_proba(test_data[feature_list]))
```

保存结果：


```python
col_names = np.sort(train['Category'].unique())
print col_names
result = pd.DataFrame(data=test_predicted, columns=col_names)
result['Id'] = test['Id'].astype(int)
result.to_csv('output.csv', index=False)
```

    ['ARSON' 'ASSAULT' 'BAD CHECKS' 'BRIBERY' 'BURGLARY' 'DISORDERLY CONDUCT'
     'DRIVING UNDER THE INFLUENCE' 'DRUG/NARCOTIC' 'DRUNKENNESS' 'EMBEZZLEMENT'
     'EXTORTION' 'FAMILY OFFENSES' 'FORGERY/COUNTERFEITING' 'FRAUD' 'GAMBLING'
     'KIDNAPPING' 'LARCENY/THEFT' 'LIQUOR LAWS' 'LOITERING' 'MISSING PERSON'
     'NON-CRIMINAL' 'OTHER OFFENSES' 'PORNOGRAPHY/OBSCENE MAT' 'PROSTITUTION'
     'RECOVERED VEHICLE' 'ROBBERY' 'RUNAWAY' 'SECONDARY CODES'
     'SEX OFFENSES FORCIBLE' 'SEX OFFENSES NON FORCIBLE' 'STOLEN PROPERTY'
     'SUICIDE' 'SUSPICIOUS OCC' 'TREA' 'TRESPASS' 'VANDALISM' 'VEHICLE THEFT'
     'WARRANTS' 'WEAPON LAWS']
