
# coding: utf-8

# ## 识别谷歌街景图片中的字母
# 
# [street-view-getting-started-with-julia](https://www.kaggle.com/c/street-view-getting-started-with-julia) 让我们从谷歌街景的图片中鉴定字母，这个题目是让我们学习和使用Julia，Julia有python和R的易用性，有C语言的速度，无奈对Julia不是很熟悉，所以还是想用python来试试。

# In[1]:

import cv2
import numpy as np
import sys
import pandas as pd


# 我们希望所有的图片最后存储在一个numpy的矩阵当中，每一行为图片的像素值。为了得到统一的表达呢，我们将RGB三个通道的值做平均得到的灰度图像作为每个图片的表示:

# In[14]:

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


# ### 预处理训练集和测试集

# In[15]:

imageSize = 400
trainlabels = pd.read_csv("trainLabels.csv")
testlabels = pd.read_csv("sampleSubmission.csv")
# 得到训练集的特征
xTrain = read_data('train', trainlabels, imageSize)
# 得到测试集的特征
xTest = read_data("test", testlabels, imageSize)


# #### 预览数据：

# In[19]:

print trainlabels.head(2)
print testlabels.head(2)


# In[20]:

yTrain = trainlabels["Class"]
yTrain = [ord(x) for x in yTrain]


# ## 模型训练
# 
# ### 随机森林
# 
# 使用随机森林进行训练，树的个数和深度需要多次调解寻求最佳值

# In[37]:

from sklearn.ensemble import RandomForestClassifier
get_ipython().magic(u'time rfc = RandomForestClassifier(n_estimators = 500, max_features = 50, max_depth=None)')
rfc.fit(xTrain, yTrain)


# #### 预测
# 将训练后的模型应用到测试集上，并保存结果：

# In[31]:

predTest = rfc.predict(xTest)
predResult = [chr(x) for x in predTest]
testlabels["Class"] = predResult
testlabels.to_csv("rf_500_50_result.csv",index = None)


# #### 结果
# 使用50颗树进行训练，提交kaggle之后准确率约为0.40  
# 改用300颗树进行训练，提交kaggle之后准确率为0.46695  
# 改用500颗树进行训练，深度为10，提价kaggle后准确率为0.40，估计出现了过拟合  
# 改用500颗树进行训练，不设置深度，提价kaggle后准确率为0.47480  

# ### 贝叶斯

# In[27]:

from sklearn.naive_bayes import GaussianNB as GNB
model_GNB = GNB()
model_GNB.fit(xTrain, yTrain)

predTest = model_GNB.predict(xTest)
predResult = [chr(x) for x in predTest]
testlabels["Class"] = predResult
testlabels.to_csv("gnb_result.csv",index = None)


# 贝叶斯的训练非常的快，把结果提交kaggle后，得到0.02389的准确率，明显低于随机森林

# ### GBDT

# In[36]:

from sklearn.ensemble import GradientBoostingClassifier
get_ipython().magic(u"time GBDT = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,                         min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None,                         random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')")

get_ipython().magic(u'time GBDT.fit(xTrain, yTrain)')

get_ipython().magic(u'time predTest = GBDT.predict(xTest)')
predResult = [chr(x) for x in predTest]
testlabels["Class"] = predResult
testlabels.to_csv("gbdt_result.csv",index = None)


# 使用GBDT仅得到了0.31937的准确率，可能是我的默认参数没有调节好，关键是GBDT的训练时间太长，调试成本也比较高

# ### 神经网络

# In[40]:

import os
from skimage.io import imread
from lasagne import layers
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet, BatchIterator


# In[44]:

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


# In[45]:

# 预处理数据，首先将图片保存为32*32的小图片
imageSize = 1024 # 32 x 32
image_width = image_height = int(imageSize ** 0.5)

labelsInfoTrain = pd.read_csv            ('trainLabels.csv'.format(path))
labelsInfoTest = pd.read_csv            ('sampleSubmission.csv'.format(path))

# Load dataset
nnxTrain = read_datax('train', labelsInfoTrain, imageSize, '.')
nnxTest = read_datax('test', labelsInfoTest, imageSize, '.')

nnyTrain = map(ord, labelsInfoTrain['Class'])
nnyTrain = np.array(yTrain)


# In[46]:

# 归一化数据
nnxTrain /= nnxTrain.std(axis = None)
nnxTrain -= nnxTrain.mean()

nnxTest /= nnxTest.std(axis = None)
nnxTest -= nnxTest.mean()


# In[47]:

# Reshape data
train_x_reshaped = nnxTrain.reshape(nnxTrain.shape[0], 1, 
                  image_height, image_width).astype('float32')
test_x_reshaped = nnxTest.reshape(nnxTest.shape[0], 1, 
                  image_height, image_width).astype('float32')


# In[54]:

# 进行训练和测试
predict = fit_model(train_x_reshaped, nnyTrain, image_width, image_height, test_x_reshaped)


# In[55]:

# 保存结果
yTest = map(chr, predict)
labelsInfoTest['Class'] = yTest
labelsInfoTest.to_csv('nnresult.csv'.format(path), index = False)


# 提交kaggle之后的准确率：0.64562
