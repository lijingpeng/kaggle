
## 数字序列预测

[Kaggle地址](https://www.kaggle.com/c/integer-sequence-learning)


```python
# -*- coding: UTF-8 -*-
%matplotlib inline

import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
```


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```


```python
last = test.Sequence.apply(lambda x: pd.Series(x.split(','))).mode(axis=1).fillna(0)

```

```python
submission = pd.DataFrame({'Id': test['Id'], 'Last': last[0]})
submission.to_csv('mode.csv', index=False)
```

提交Kaggle之后是0.05680
