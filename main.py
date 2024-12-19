import os
import math
import time
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import  matplotlib
from dataset import *
from train import *
from test import *

from 深度学习课设.text import Test
#数据处理
getdata = GetData(train_path='data/train.csv',test_path='data/testA.csv')
getdata.data_processing()
#对训练数据和测试数据格式化
getdata.convert_data()
print(getdata.train_data.shape)
print(getdata.train_label.shape)
print(getdata.test_data.shape)


getdata.train_data.head()
getdata.train_data.describe()
getdata.train_data.info()
plt.hist(getdata.train['label'], orientation = 'vertical', histtype = 'bar', color = 'red')
plt.show()

category0 = getdata.train[getdata.train["label"] == 0].values[:, 1:-1]
category1 = getdata.train[getdata.train["label"] == 1].values[:, 1:-1]
category2 = getdata.train[getdata.train["label"] == 2].values[:, 1:-1]
category3 = getdata.train[getdata.train["label"] == 3].values[:, 1:-1]
plt.figure(figsize=(30, 5))
plt.subplot(1, 4, 1)
for i in range(5):
    plt.plot(category0[i])
plt.subplot(1, 4, 2)
for i in range(5):
    plt.plot(category1[i])
plt.subplot(1, 4, 3)
for i in range(5):
    plt.plot(category2[i])
plt.subplot(1, 4, 4)
for i in range(5):
    plt.plot(category3[i])

plt.cla()
plt.plot(category0.mean(axis=0), label="category0")
plt.plot(category1.mean(axis=0), label="category1")
plt.plot(category2.mean(axis=0), label="category2")
plt.plot(category3.mean(axis=0), label="category3")
plt.legend()

train = Train(
    train_path="data/train.csv",
    test_path="data/testA.csv",
    result_path="result",
    num_classes=4,
    growth_rate=32,
    block_config=[6, 12, 64, 48],
    num_init_features=64,
    bn_size=4,
    drop_rate=0,
    num_epochs=50,
    batch_size=250,
    lr=1e-1,
    weight_decay=1e-4,
    device="cpu",
    resume=False,
)
train.setup_seed(2021)
train.dataload()
train.build_model()
train.define_loss()
train.define_optim()
train.train()
print("training finished!")


test = Test(
    train_path="data/train.csv",
    test_path="data/testA.csv",
    result_path="result",
    num_classes=4,
    batch_size=200,
    block_config=[6, 12, 64, 48],
    device="cpu",
)
test.setup_seed(2022)
test.dataload()
test.build_model()
test.test()