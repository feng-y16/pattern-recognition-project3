import sys
sys.path.append("F:/api/libsvm/python")
from svm import *
from svmutil import *
import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

#r=np.load("10genes_withoutnorm.npz")
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
index=range(0,10)#shuffle
train_total=shuffledata(train_data,train_label,train_size,10)
train_data=train_total[:,[0,4,6]]
train_label=train_total[:,10]
#######################################################################################################
times=5000#交叉验证
score=0
for i in range(0,times):
    acc = svm_train(train_label,train_data,'-q -t 0 -v 10')#线性核函数
    score=score+acc/100
score=score/times
print("average:",end="")
print(score)
#J1_2features:0.9529297002725104
#J1_3features:0.9505204359673417
#H2_2features:0.8220103542234029
#H2_3features:0.8992991825613114
#4+6:0.965677929155361
#0+4+6:0.9721787465939833