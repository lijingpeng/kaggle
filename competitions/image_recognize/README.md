
## 识别谷歌街景图片中的字母

[street-view-getting-started-with-julia](https://www.kaggle.com/c/street-view-getting-started-with-julia) 让我们从谷歌街景的图片中鉴定字母，这个题目是让我们学习和使用Julia，Julia有python和R的易用性，有C语言的速度，无奈对Julia不是很熟悉，所以还是想用python来试试。


```python
import cv2
import numpy as np
import sys
import pandas as pd
```

我们希望所有的图片最后存储在一个numpy的矩阵当中，每一行为图片的像素值。为了得到统一的表达呢，我们将RGB三个通道的值做平均得到的灰度图像作为每个图片的表示:


```python
# typeData 为"train"或者"test"
# labelsInfo 包含每一个图片的ID
# 图片存储在trainResized和testResized文件夹内
def read_data(typeData, labelsInfo, imageSize):
    labelsIndex = labelsInfo["ID"]
    x = np.zeros((np.size(labelsIndex), imageSize))
    for idx, idImage in enumerate(labelsIndex):
        # 得到图片文件名并读取
        nameFile = typeData + "Resized/" + str(idImage) + ".Bmp"
        img = cv2.imread(nameFile)
        # 转化为灰度图
        temp = np.mean(img, 2)
        # 将图片转化为行向量
        x[idx, :] = np.reshape(temp, (1, imageSize))
    return x
```

### 预处理训练集和测试集


```python
imageSize = 400
trainlabels = pd.read_csv("trainLabels.csv")
testlabels = pd.read_csv("sampleSubmission.csv")
# 得到训练集的特征
xTrain = read_data('train', trainlabels, imageSize)
# 得到测试集的特征
xTest = read_data("test", testlabels, imageSize)
```

#### 预览数据：


```python
print trainlabels.head(2)
print testlabels.head(2)
```

       ID Class
    0   1     n
    1   2     8
         ID Class
    0  6284     A
    1  6285     A



```python
yTrain = trainlabels["Class"]
yTrain = [ord(x) for x in yTrain]
```

## 模型训练

### 随机森林

使用随机森林进行训练，树的个数和深度需要多次调解寻求最佳值


```python
from sklearn.ensemble import RandomForestClassifier
%time rfc = RandomForestClassifier(n_estimators = 500, max_features = 50, max_depth=None)
rfc.fit(xTrain, yTrain)
```

    CPU times: user 121 µs, sys: 367 µs, total: 488 µs
    Wall time: 494 µs





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features=50, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



#### 预测
将训练后的模型应用到测试集上，并保存结果：


```python
predTest = rfc.predict(xTest)
predResult = [chr(x) for x in predTest]
testlabels["Class"] = predResult
testlabels.to_csv("rf_500_50_result.csv",index = None)
```

#### 结果
使用50颗树进行训练，提交kaggle之后准确率约为0.40  
改用300颗树进行训练，提交kaggle之后准确率为0.46695  
改用500颗树进行训练，深度为10，提价kaggle后准确率为0.40，估计出现了过拟合  
改用500颗树进行训练，不设置深度，提价kaggle后准确率为0.47480  

### 贝叶斯


```python
from sklearn.naive_bayes import GaussianNB as GNB
model_GNB = GNB()
model_GNB.fit(xTrain, yTrain)

predTest = model_GNB.predict(xTest)
predResult = [chr(x) for x in predTest]
testlabels["Class"] = predResult
testlabels.to_csv("gnb_result.csv",index = None)
```

贝叶斯的训练非常的快，把结果提交kaggle后，得到0.02389的准确率，明显低于随机森林

### GBDT


```python
from sklearn.ensemble import GradientBoostingClassifier
%time GBDT = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, \
                        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, \
                        random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

%time GBDT.fit(xTrain, yTrain)

%time predTest = GBDT.predict(xTest)
predResult = [chr(x) for x in predTest]
testlabels["Class"] = predResult
testlabels.to_csv("gbdt_result.csv",index = None)
```

    CPU times: user 91 µs, sys: 738 µs, total: 829 µs
    Wall time: 2.93 ms
    CPU times: user 40min 16s, sys: 52.3 s, total: 41min 9s
    Wall time: 2h 55min 22s
    CPU times: user 1.75 s, sys: 44.5 ms, total: 1.8 s
    Wall time: 1.79 s


使用GBDT仅得到了0.31937的准确率，可能是我的默认参数没有调节好，关键是GBDT的训练时间太长，调试成本也比较高

### 神经网络


```python
import os
from skimage.io import imread
from lasagne import layers
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet, BatchIterator
```


```python
# Define functions
def read_datax(typeData, labelsInfo, imageSize, path):
    x = np.zeros((labelsInfo.shape[0], imageSize))
    
    for (index, idImage) in enumerate(labelsInfo['ID']):
        # use specially created 32 x 32 images
        nameFile = '{0}/{1}Resized32/{2}.Bmp'.format(path, 
                    typeData, idImage)
        img = imread(nameFile, as_grey = True)
        
        x[index, :] = np.reshape(img, (1, imageSize))
        
    return x

def fit_model(reshaped_train_x, y, image_width, 
                    image_height, reshaped_test_x):
    net = NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('hidden4', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape = (None, 1, 32, 32),
        conv1_num_filters=32, conv1_filter_size=(5, 5), 
        pool1_pool_size=(2, 2),
        dropout1_p=0.2,
        conv2_num_filters=64, conv2_filter_size=(5, 5), 
        pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters = 128, conv3_filter_size = (5, 5),
        hidden4_num_units=500,
        output_num_units = 62, output_nonlinearity = softmax,
        
        update_learning_rate = 0.01,
        update_momentum = 0.9,
        
        batch_iterator_train = BatchIterator(batch_size = 100),
        batch_iterator_test = BatchIterator(batch_size = 100),
        
        use_label_encoder = True,
        regression = False,
        max_epochs = 100,
        verbose = 1,
    )
    
    net.fit(reshaped_train_x, y)
    prediction = net.predict(reshaped_test_x)
    
    return prediction
```


```python
# 预处理数据，首先将图片保存为32*32的小图片
imageSize = 1024 # 32 x 32
image_width = image_height = int(imageSize ** 0.5)

labelsInfoTrain = pd.read_csv\
            ('trainLabels.csv'.format(path))
labelsInfoTest = pd.read_csv\
            ('sampleSubmission.csv'.format(path))

# Load dataset
nnxTrain = read_datax('train', labelsInfoTrain, imageSize, '.')
nnxTest = read_datax('test', labelsInfoTest, imageSize, '.')

nnyTrain = map(ord, labelsInfoTrain['Class'])
nnyTrain = np.array(yTrain)
```


```python
# 归一化数据
nnxTrain /= nnxTrain.std(axis = None)
nnxTrain -= nnxTrain.mean()

nnxTest /= nnxTest.std(axis = None)
nnxTest -= nnxTest.mean()
```


```python
# Reshape data
train_x_reshaped = nnxTrain.reshape(nnxTrain.shape[0], 1, 
                  image_height, image_width).astype('float32')
test_x_reshaped = nnxTest.reshape(nnxTest.shape[0], 1, 
                  image_height, image_width).astype('float32')
```


```python
# 进行训练和测试
predict = fit_model(train_x_reshaped, nnyTrain, image_width, image_height, test_x_reshaped)
```

    # Neural Network with 352586 learnable parameters
    
    ## Layer information
    
      #  name      size
    ---  --------  --------
      0  input     1x32x32
      1  conv1     32x28x28
      2  pool1     32x14x14
      3  dropout1  32x14x14
      4  conv2     64x10x10
      5  pool2     64x5x5
      6  dropout2  64x5x5
      7  conv3     128x1x1
      8  hidden4   500
      9  output    62
    
      epoch    trn loss    val loss    trn/val    valid acc  dur
    -------  ----------  ----------  ---------  -----------  ------
          1     [36m4.08201[0m     [32m4.01012[0m    1.01793      0.07254  16.55s
          2     [36m3.87688[0m     [32m3.84326[0m    1.00875      0.04836  17.72s
          3     [36m3.82788[0m     [32m3.79976[0m    1.00740      0.04914  16.58s
          4     [36m3.78741[0m     [32m3.78872[0m    0.99965      0.07254  16.14s
          5     [36m3.78030[0m     [32m3.78600[0m    0.99850      0.07254  16.37s
          6     [36m3.77679[0m     [32m3.78520[0m    0.99778      0.07254  16.56s
          7     [36m3.77487[0m     3.78537    0.99723      0.07254  16.30s
          8     [36m3.77411[0m     [32m3.78468[0m    0.99721      0.07254  16.51s
          9     [36m3.77257[0m     3.78518    0.99667      0.07254  15.92s
         10     [36m3.77202[0m     [32m3.78459[0m    0.99668      0.07254  16.55s
         11     [36m3.76948[0m     [32m3.78458[0m    0.99601      0.07254  16.25s
         12     [36m3.76882[0m     [32m3.78414[0m    0.99595      0.07254  16.31s
         13     [36m3.76717[0m     [32m3.78411[0m    0.99552      0.07254  15.70s
         14     [36m3.76606[0m     3.78469    0.99508      0.07254  16.04s
         15     [36m3.76419[0m     3.78671    0.99405      0.07176  15.70s
         16     [36m3.76277[0m     [32m3.78392[0m    0.99441      0.07176  16.05s
         17     [36m3.76014[0m     3.78821    0.99259      0.07176  15.71s
         18     3.78179     3.78606    0.99887      0.07254  16.11s
         19     3.76928     [32m3.78321[0m    0.99632      0.07254  15.75s
         20     3.76688     3.78358    0.99559      0.07254  16.05s
         21     3.76434     [32m3.78255[0m    0.99519      0.07254  17.36s
         22     3.76186     [32m3.78174[0m    0.99474      0.07254  18.12s
         23     [36m3.75829[0m     3.78184    0.99377      0.07878  17.90s
         24     [36m3.75370[0m     3.78545    0.99161      0.07488  18.19s
         25     [36m3.74749[0m     [32m3.77908[0m    0.99164      0.07098  17.81s
         26     [36m3.73650[0m     [32m3.77806[0m    0.98900      0.07020  18.08s
         27     [36m3.71592[0m     [32m3.77626[0m    0.98402      0.06474  18.03s
         28     [36m3.67805[0m     [32m3.74531[0m    0.98204      0.07176  18.04s
         29     [36m3.59550[0m     3.79802    0.94668      0.07566  18.12s
         30     [36m3.44086[0m     [32m3.35483[0m    1.02564      0.19111  18.06s
         31     [36m3.14160[0m     [32m3.00021[0m    1.04713      0.29251  17.41s
         32     [36m2.73389[0m     [32m2.89130[0m    0.94556      0.31903  16.19s
         33     [36m2.61587[0m     [32m2.53098[0m    1.03354      0.38144  15.73s
         34     [36m2.25316[0m     [32m2.26086[0m    0.99660      0.43994  16.14s
         35     [36m1.95499[0m     [32m2.03661[0m    0.95993      0.48206  15.76s
         36     [36m1.75483[0m     [32m1.94987[0m    0.89997      0.49610  16.01s
         37     [36m1.60276[0m     [32m1.78637[0m    0.89722      0.52106  15.60s
         38     [36m1.47862[0m     [32m1.73524[0m    0.85211      0.54524  15.98s
         39     [36m1.35049[0m     [32m1.65705[0m    0.81500      0.55694  15.62s
         40     [36m1.27458[0m     [32m1.65253[0m    0.77129      0.57254  16.01s
         41     [36m1.18548[0m     [32m1.60550[0m    0.73839      0.58112  15.61s
         42     [36m1.11862[0m     1.62259    0.68940      0.58268  16.51s
         43     [36m1.05698[0m     1.68044    0.62899      0.58112  16.24s
         44     [36m1.01350[0m     1.64642    0.61558      0.59126  16.50s
         45     [36m0.93587[0m     1.62059    0.57749      0.59906  15.81s
         46     [36m0.87893[0m     1.65983    0.52953      0.59984  16.54s
         47     [36m0.83695[0m     1.66309    0.50325      0.60452  16.42s
         48     1.72887     2.92194    0.59169      0.54446  16.31s
         49     3.85830     3.39520    1.13640      0.21373  15.84s
         50     2.26598     1.97743    1.14592      0.46724  18.41s
         51     2.11105     1.89927    1.11150      0.49298  18.02s
         52     1.66393     1.75705    0.94700      0.51794  17.99s
         53     1.48332     1.65795    0.89467      0.54212  17.94s
         54     1.38197     [32m1.60296[0m    0.86214      0.55928  17.73s
         55     1.28419     [32m1.56050[0m    0.82293      0.56318  17.94s
         56     1.21078     [32m1.54983[0m    0.78123      0.57176  17.70s
         57     1.13885     1.55330    0.73318      0.55616  17.93s
         58     1.10488     [32m1.53462[0m    0.71997      0.57956  17.71s
         59     1.03479     1.54234    0.67092      0.58502  17.70s
         60     0.98439     [32m1.52492[0m    0.64554      0.59984  17.95s
         61     0.93277     [32m1.49128[0m    0.62548      0.59204  17.67s
         62     1.03055     1.58280    0.65109      0.57878  18.01s
         63     0.89008     1.54904    0.57460      0.59750  17.69s
         64     0.83698     1.59463    0.52487      0.58346  17.92s
         65     [36m0.79801[0m     1.59534    0.50021      0.60452  17.80s
         66     [36m0.77752[0m     1.56702    0.49618      0.60842  17.91s
         67     [36m0.73901[0m     1.61821    0.45668      0.59594  17.81s
         68     [36m0.71108[0m     1.56703    0.45377      0.61154  17.98s
         69     [36m0.67279[0m     1.61497    0.41659      0.61154  17.81s
         70     [36m0.64651[0m     1.66452    0.38841      0.60530  17.97s
         71     [36m0.61597[0m     1.65828    0.37145      0.62012  17.84s
         72     [36m0.59188[0m     1.69796    0.34858      0.60296  17.92s
         73     [36m0.57862[0m     1.72392    0.33564      0.60686  17.73s
         74     [36m0.56451[0m     1.75449    0.32175      0.60062  17.56s
         75     [36m0.53835[0m     1.74351    0.30877      0.62090  17.77s
         76     [36m0.53288[0m     1.80642    0.29499      0.60842  18.08s
         77     [36m0.49975[0m     1.76941    0.28244      0.61700  17.76s
         78     [36m0.48489[0m     1.75930    0.27561      0.60998  17.92s
         79     [36m0.45688[0m     1.81943    0.25111      0.61622  17.78s
         80     0.46801     1.80187    0.25974      0.62480  17.96s
         81     [36m0.45527[0m     1.88136    0.24199      0.61310  17.84s
         82     [36m0.43178[0m     1.93961    0.22261      0.61622  18.56s
         83     [36m0.41726[0m     1.90341    0.21922      0.61856  16.52s
         84     [36m0.38590[0m     1.91029    0.20201      0.61778  15.59s
         85     [36m0.38510[0m     1.93524    0.19900      0.61778  16.00s
         86     [36m0.37565[0m     1.92514    0.19513      0.61466  15.56s
         87     [36m0.36222[0m     1.99870    0.18123      0.61544  15.88s
         88     0.38495     2.08839    0.18433      0.61466  15.55s
         89     [36m0.34101[0m     1.94872    0.17499      0.62559  15.97s
         90     [36m0.33575[0m     2.01506    0.16662      0.61856  15.63s
         91     [36m0.32353[0m     2.05956    0.15709      0.62090  16.03s
         92     [36m0.30422[0m     2.12548    0.14313      0.64041  15.66s
         93     [36m0.29631[0m     2.10645    0.14067      0.63495  16.02s
         94     0.32050     2.11861    0.15128      0.62168  15.73s
         95     0.30140     2.14516    0.14050      0.62871  15.99s
         96     [36m0.28195[0m     2.09292    0.13472      0.63339  15.67s
         97     0.30323     2.20744    0.13737      0.62246  16.07s
         98     [36m0.27107[0m     2.15645    0.12570      0.63729  16.32s
         99     0.27947     2.22565    0.12557      0.62637  16.51s
        100     [36m0.26500[0m     2.22825    0.11893      0.64431  16.52s



```python
# 保存结果
yTest = map(chr, predict)
labelsInfoTest['Class'] = yTest
labelsInfoTest.to_csv('nnresult.csv'.format(path), index = False)
```

提交kaggle之后的准确率：0.64562
