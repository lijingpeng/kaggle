
## 电影文本情感分类

[Github地址](https://github.com/lijingpeng/kaggle/tree/master/competitions/Bag_of_Words)  
[Kaggle地址](https://www.kaggle.com/c/word2vec-nlp-tutorial/)

这个任务主要是对电影评论文本进行情感分类，主要分为正面评论和负面评论，所以是一个二分类问题，二分类模型我们可以选取一些常见的模型比如贝叶斯、逻辑回归等，这里挑战之一是文本内容的向量化，因此，我们首先尝试基于TF-IDF的向量化方法，然后尝试word2vec。


```python
# -*- coding: UTF-8 -*-
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
train = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/Bag_of_Words_Meets_Bags_of_Popcorn/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('/Users/frank/Documents/workspace/kaggle/dataset/Bag_of_Words_Meets_Bags_of_Popcorn/testData.tsv', header=0, delimiter="\t", quoting=3)
print train.head()
print test.head()
```

             id  sentiment                                             review
    0  "5814_8"          1  "With all this stuff going down at the moment ...
    1  "2381_9"          1  "\"The Classic War of the Worlds\" by Timothy ...
    2  "7759_3"          0  "The film starts with a manager (Nicholas Bell...
    3  "3630_4"          0  "It must be assumed that those who praised thi...
    4  "9495_8"          1  "Superbly trashy and wondrously unpretentious ...
               id                                             review
    0  "12311_10"  "Naturally in a film who's main themes are of ...
    1    "8348_2"  "This movie is a disaster within a disaster fi...
    2    "5828_4"  "All in all, this is a movie for kids. We saw ...
    3    "7186_2"  "Afraid of the Dark left me with the impressio...
    4   "12128_7"  "A very accurate depiction of small time mob l...


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
tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
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

    多项式贝叶斯分类器10折交叉验证得分:  0.94983968



```python
test_predicted = np.array(model_NB.predict(test_x))
print '保存结果...'
nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
nb_output['id'] = test['id']
nb_output = nb_output[['id', 'sentiment']]
nb_output.to_csv('nb_output.csv', index=False)
print '结束.'
```

    保存结果...
    结束.


1. 提交最终的结果到kaggle，AUC为：0.85728，排名300左右，50%的水平  
2. ngram_range = 3, 三元文法，AUC为0.85924

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

    [mean: 0.96497, std: 0.00476, params: {'C': 30}]



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


1. 提交最终的结果到kaggle，AUC为：0.88956，排名260左右，比之前贝叶斯模型有所提高   
2. 三元文法，AUC为0.89076

## Word2vec
神经网络语言模型L = SUM[log(p(w|contect(w))]，即在w的上下文下计算当前词w的概率，由公式可以看到，我们的核心是计算p(w|contect(w)， Word2vec给出了构造这个概率的一个方法。


```python
import gensim
import nltk
from nltk.corpus import stopwords

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    '''
    将评论段落转换为句子，返回句子列表，每个句子由一堆词组成
    '''
    raw_sentences = tokenizer.tokenize(review.strip().decode('utf8'))

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 获取句子中的词列表
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences
```


```python
sentences = []
for i, review in enumerate(train["review"]):
    sentences += review_to_sentences(review, tokenizer)
```


```python
unlabeled_train = pd.read_csv("/Users/frank/Documents/workspace/kaggle/dataset/Bag_of_Words_Meets_Bags_of_Popcorn/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print '预处理unlabeled_train data...'
print len(train_data)
print len(sentences)
```

```
    预处理unlabeled_train data...
    25000
    795538
```

### 构建word2vec模型


```python
import time
from gensim.models import Word2Vec
# 模型参数
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
```


```python
%%time
# 训练模型
print("训练模型中...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
```

    训练模型中...
    CPU times: user 6min 16s, sys: 8.34 s, total: 6min 24s
    Wall time: 2min 27s



```python
print '保存模型...'
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)
```

    保存模型...


### 预览模型


```python
model.doesnt_match("man woman child kitchen".split())
```




    'kitchen'




```python
model.doesnt_match("france england germany berlin".split())
```




    'berlin'




```python
model.doesnt_match("paris berlin london austria".split())
```




    'london'




```python
model.most_similar("man")
```




    [(u'woman', 0.6246455907821655),
     (u'lady', 0.6008599400520325),
     (u'lad', 0.5698915719985962),
     (u'businessman', 0.5431989431381226),
     (u'chap', 0.53116375207901),
     (u'monk', 0.5250570774078369),
     (u'men', 0.5177899599075317),
     (u'guy', 0.517480731010437),
     (u'farmer', 0.5114585757255554),
     (u'person', 0.5109285116195679)]




```python
model.most_similar("queen")
```




    [(u'princess', 0.6759523153305054),
     (u'bride', 0.6207793951034546),
     (u'belle', 0.6001157760620117),
     (u'shearer', 0.5995810031890869),
     (u'stepmother', 0.596365749835968),
     (u'victoria', 0.5917614698410034),
     (u'dame', 0.589063286781311),
     (u'latifah', 0.5790275931358337),
     (u'countess', 0.5776904821395874),
     (u'widow', 0.5727116465568542)]




```python
model.most_similar("awful")
```




    [(u'terrible', 0.7642339468002319),
     (u'atrocious', 0.7405279874801636),
     (u'horrible', 0.7376815676689148),
     (u'abysmal', 0.7010303139686584),
     (u'dreadful', 0.6942194104194641),
     (u'appalling', 0.6887971758842468),
     (u'lousy', 0.6646767854690552),
     (u'horrid', 0.6554058194160461),
     (u'horrendous', 0.6533403992652893),
     (u'amateurish', 0.6079087853431702)]



### 使用Word2vec特征


```python
def makeFeatureVec(words, model, num_features):
    '''
    对段落中的所有词向量进行取平均操作
    '''
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # 取平均
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0.

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
       if counter % 5000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))

       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)

       counter = counter + 1.
    return reviewFeatureVecs
```


```python
%time trainDataVecs = getAvgFeatureVecs( train_data, model, num_features )
```

    Review 0 of 25000
    Review 5000 of 25000
    Review 10000 of 25000
    Review 15000 of 25000
    Review 20000 of 25000
    CPU times: user 1min 49s, sys: 1.9 s, total: 1min 51s
    Wall time: 1min 54s



```python
%time testDataVecs = getAvgFeatureVecs(test_data, model, num_features)
```

    Review 0 of 25000
    Review 5000 of 25000
    Review 10000 of 25000
    Review 15000 of 25000
    Review 20000 of 25000
    CPU times: user 1min 44s, sys: 1.56 s, total: 1min 46s
    Wall time: 1min 48s


### 高斯贝叶斯+Word2vec训练


```python
from sklearn.naive_bayes import GaussianNB as GNB

model_GNB = GNB()
model_GNB.fit(trainDataVecs, label)

from sklearn.cross_validation import cross_val_score
import numpy as np

print "高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, trainDataVecs, label, cv=10, scoring='roc_auc'))

result = forest.predict( testDataVecs )

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "gnb_word2vec.csv", index=False, quoting=3 )
```

    多项式贝叶斯分类器10折交叉验证得分:  0.625579296


从验证结果来看，没有超过基于TF-IDF多项式贝叶斯模型

### 随机森林+Word2vec训练


```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)

print("Fitting a random forest to labeled training data...")
%time forest = forest.fit( trainDataVecs, label )
print "随机森林分类器10折交叉验证得分: ", np.mean(cross_val_score(forest, trainDataVecs, label, cv=10, scoring='roc_auc'))

# 测试集
result = forest.predict( testDataVecs )

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "rf_word2vec.csv", index=False, quoting=3 )
```

    Fitting a random forest to labeled training data...
    CPU times: user 45 s, sys: 460 ms, total: 45.5 s
    Wall time: 24.2 s
    随机森林分类器10折交叉验证得分:  0.648426368


改用随机森林之后，效果有提升，但是依然没有超过基于TF-IDF多项式贝叶斯模型
