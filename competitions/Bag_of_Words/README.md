
## 电影文本情感分类


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

           id  sentiment                                             review
    0  5814_8          1  With all this stuff going down at the moment w...
    1  2381_9          1  \The Classic War of the Worlds\" by Timothy Hi...
    2  7759_3          0  The film starts with a manager (Nicholas Bell)...
    3  3630_4          0  It must be assumed that those who praised this...
    4  9495_8          1  Superbly trashy and wondrously unpretentious 8...
             id                                             review
    0  12311_10  Naturally in a film who's main themes are of m...
    1    8348_2  This movie is a disaster within a disaster fil...
    2    5828_4  All in all, this is a movie for kids. We saw i...
    3    7186_2  Afraid of the Dark left me with the impression...
    4   12128_7  A very accurate depiction of small time mob li...


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

    with all this stuff going down at the moment with mj i ve started listening to his music watching the odd documentary here and there watched the wiz and watched moonwalker again maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent moonwalker is part biography part feature film which i remember going to see at the cinema when it was originally released some of it has subtle messages about mj s feeling towards the press and also the obvious message of drugs are bad m kay visually impressive but of course this is all about michael jackson so unless you remotely like mj in anyway then you are going to hate this and find it boring some may call mj an egotist for consenting to the making of this movie but mj and most of his fans would say that he made it for the fans which if true is really nice of him the actual feature film bit when it finally starts is only on for minutes or so excluding the smooth criminal sequence and joe pesci is convincing as a psychopathic all powerful drug lord why he wants mj dead so bad is beyond me because mj overheard his plans nah joe pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno maybe he just hates mj s music lots of cool things in this like mj turning into a car and a robot and the whole speed demon sequence also the director must have had the patience of a saint when it came to filming the kiddy bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene bottom line this movie is for people who like mj on one level or another which i think is most people if not then stay away it does try and give off a wholesome message and ironically mj s bestest buddy in this movie is a girl michael jackson is truly one of the most talented people ever to grace this planet but is he guilty well with all the attention i ve gave this subject hmmm well i don t know because people can be different behind closed doors i know this for a fact he is either an extremely nice but stupid guy or one of the most sickest liars i hope he is not the latter 
    
    naturally in a film who s main themes are of mortality nostalgia and loss of innocence it is perhaps not surprising that it is rated more highly by older viewers than younger ones however there is a craftsmanship and completeness to the film which anyone can enjoy the pace is steady and constant the characters full and engaging the relationships and interactions natural showing that you do not need floods of tears to show emotion screams to show fear shouting to show dispute or violence to show anger naturally joyce s short story lends the film a ready made structure as perfect as a polished diamond but the small changes huston makes such as the inclusion of the poem fit in neatly it is truly a masterpiece of tact subtlety and overwhelming beauty


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

    TF-IDF处理结束.


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

    多项式贝叶斯分类器10折交叉验证得分:  0.949135808



```python
test_predicted = np.array(model_NB.predict(test_x))
print '保存结果...'
nb_output['id'] = test['id']
nb_output['sentiment'] = pd.DataFrame(data=test_predicted)
nb_output.to_csv('nb_output.csv', index=False)
print '结束.'
```

    保存结果...
    结束.


提交最终的结果到kaggle，准确率为：0.85728，排名300左右，50%的水平

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

    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)
    /Library/Python/2.7/site-packages/sklearn/svm/base.py:874: DeprecationWarning: penalty='L2' has been deprecated in favor of penalty='l2' as of 0.16. Backward compatibility for the uppercase notation will be removed in 0.18
      DeprecationWarning)


    [mean: 0.96461, std: 0.00490, params: {'C': 30}]



```python
test_predicted = np.array(model_LR.predict(test_x))
print '保存结果...'
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment']]
lr_output.to_csv('lr_output.csv', index=False)
print '结束.'
```

    保存结果...
    结束.


提交最终的结果到kaggle，准确率为：0.88956，排名260左右，比之前贝叶斯模型有所提高
