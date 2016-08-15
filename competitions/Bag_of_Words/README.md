
## 电影文本情感分类

[Github地址](https://github.com/lijingpeng/kaggle/tree/master/competitions/Bag_of_Words)  
[Kaggle地址](https://www.kaggle.com/c/word2vec-nlp-tutorial/)

这个任务主要是对电影评论文本进行情感分类，主要分为正面评论和负面评论，所以是一个二分类问题，二分类模型我们可以选取一些常见的模型比如贝叶斯、逻辑回归等，这里挑战之一是文本内容的向量化，因此，我们首先尝试基于TF-IDF的向量化方法，然后尝试word2vec。


```python
import pandas as pd
import numpy as np
import re 
from bs4 import BeautifulSoup

def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words
```

## 载入数据集


```python
# 载入数据集
train = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/Bag_of_Words_Meets_Bags_of_Popcorn/labeledTrainData.tsv', header=0, delimiter="\t")
test = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/Bag_of_Words_Meets_Bags_of_Popcorn/testData.tsv', header=0, delimiter="\t")
print train.head()
print test.head()
```

## 预处理数据


```python
# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print train_data[0], '\n'
print test_data[0]
```

## 特征处理
直接丢给计算机这些词文本，计算机是无法计算的，因此我们需要把文本转换为向量，有几种常见的文本向量处理方法，比如：
1. 单词计数  
2. TF-IDF向量  
3. Word2vec向量  
我们先使用TF-IDF来试一下。


```python
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
# 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
tfidf = TFIDF(min_df=3, # 最小支持度为3
           max_features=None, 
           strip_accents='unicode', 
           analyzer='word',
           token_pattern=r'\w{1,}', 
           ngram_range=(1, 2),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1, 
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train] 
test_x = data_all[len_train:]
print 'TF-IDF处理结束.'
```

## 朴素贝叶斯训练


```python
from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(train_x, label) 
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.cross_validation import cross_val_score
import numpy as np

print "多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc'))
```


```python
test_predicted = np.array(model_NB.predict(test_x))
print '保存结果...'
nb_output['id'] = test['id']
nb_output['sentiment'] = pd.DataFrame(data=test_predicted)
nb_output.to_csv('nb_output.csv', index=False)
print '结束.'
```

提交最终的结果到kaggle，AUC为：0.85728，排名300左右，50%的水平

## 逻辑回归


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV

# 设定grid search的参数
grid_values = {'C':[30]}  
# 设定打分为roc_auc
model_LR = GridSearchCV(LR(penalty = 'L2', dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20) 
model_LR.fit(train_x, label)
# 20折交叉验证
GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True, 
             fit_intercept=True, intercept_scaling=1, penalty='L2', random_state=0, tol=0.0001),
        fit_params={}, iid=True, n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
        scoring='roc_auc', verbose=0)
#输出结果
print model_LR.grid_scores_
```


```python
test_predicted = np.array(model_LR.predict(test_x))
print '保存结果...'
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment']]
lr_output.to_csv('lr_output.csv', index=False)
print '结束.'
```

提交最终的结果到kaggle，AUC为：0.88956，排名260左右，比之前贝叶斯模型有所提高

## Word2vec
神经网络语言模型L = SUM[log(p(w|contect(w))]，即在w的上下文下计算当前词w的概率，由公式可以看到，我们的核心是计算p(w|contect(w)， Word2vec给出了构造这个概率的一个方法。


```python
import gensim
model = gensim.models.Word2Vec(train_data, min_count=3, workers=4)
```


```python

```
