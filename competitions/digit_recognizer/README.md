
[kaggle地址](https://www.kaggle.com/c/digit-recognizer/)  

## 数据预览

首先载入数据集


```python
import pandas as pd
import numpy as np

train = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/digit_recognizer/train.csv')
test = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/digit_recognizer/test.csv')
print train.head()
print test.head()
```

       label  pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  \
    0      1       0       0       0       0       0       0       0       0   
    1      0       0       0       0       0       0       0       0       0   
    2      1       0       0       0       0       0       0       0       0   
    3      4       0       0       0       0       0       0       0       0   
    4      0       0       0       0       0       0       0       0       0   
    
       pixel8    ...     pixel774  pixel775  pixel776  pixel777  pixel778  \
    0       0    ...            0         0         0         0         0   
    1       0    ...            0         0         0         0         0   
    2       0    ...            0         0         0         0         0   
    3       0    ...            0         0         0         0         0   
    4       0    ...            0         0         0         0         0   
    
       pixel779  pixel780  pixel781  pixel782  pixel783  
    0         0         0         0         0         0  
    1         0         0         0         0         0  
    2         0         0         0         0         0  
    3         0         0         0         0         0  
    4         0         0         0         0         0  
    
    [5 rows x 785 columns]
       pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  pixel8  \
    0       0       0       0       0       0       0       0       0       0   
    1       0       0       0       0       0       0       0       0       0   
    2       0       0       0       0       0       0       0       0       0   
    3       0       0       0       0       0       0       0       0       0   
    4       0       0       0       0       0       0       0       0       0   
    
       pixel9    ...     pixel774  pixel775  pixel776  pixel777  pixel778  \
    0       0    ...            0         0         0         0         0   
    1       0    ...            0         0         0         0         0   
    2       0    ...            0         0         0         0         0   
    3       0    ...            0         0         0         0         0   
    4       0    ...            0         0         0         0         0   
    
       pixel779  pixel780  pixel781  pixel782  pixel783  
    0         0         0         0         0         0  
    1         0         0         0         0         0  
    2         0         0         0         0         0  
    3         0         0         0         0         0  
    4         0         0         0         0         0  
    
    [5 rows x 784 columns]


分离训练数据和标签：


```python
train_data = train.values[:,1:]
label = train.ix[:,0]
test_data = test.values
```

使用PCA来降维：[PCA文档](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
使用SVM来训练：[SVM文档](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## 降维


```python
from sklearn.decomposition import PCA
from sklearn.svm import SVC
pca = PCA(n_components=0.8, whiten=True)
# pca.fit(train_data)
train_data = pca.fit_transform(train_data)
# pca.fit(test_data)
test_data = pca.transform(test_data)
```

## SVM训练


```python
print('使用SVM进行训练...')
svc = SVC(kernel='rbf',C=2)
svc.fit(train_data, label)
print('训练结束.')
```

    使用SVM进行训练...
    训练结束.



```python
print('对测试集进行预测...')
predict = svc.predict(test_data)
print('预测结束.')
```

    对测试集进行预测...
    预测结束.


保存结果：


```python
pd.DataFrame(
    {"ImageId": range(1, len(predict) + 1), "Label": predict}
).to_csv('output.csv', index=False, header=True)

print 'done.'
```

    done.

