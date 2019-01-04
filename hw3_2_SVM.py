import sys
sys.path.append("F:/api/libsvm/python")
from svm import *
from svmutil import *
import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

#r=np.load("10genes_withoutnorm.npz")
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_7"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
datatemp=np.zeros((train_size+test_size,10))#shuffle
labeltemp=np.zeros(train_size+test_size)
datatemp[range(0,train_size),:]=train_data
datatemp[range(train_size,train_size+test_size),:]=test_data
labeltemp[range(0,train_size)]=train_label
labeltemp[range(train_size,train_size+test_size)]=test_label
total=shuffledata(datatemp,labeltemp,train_size+test_size,10)
index=range(0,10)
data=total[:,index]
label=total[:,10]
#######################################################################################################
model = svm_train(label,data,'-t 0 -c 0.2')#线性核函数
print("Training data predict:")
predict=svm_predict(label,data,model)
#######################################################################################################
alpha = matrix(model.get_sv_coef())#获取系数
SVs=matrix(model.get_SV())
w=matrix(np.zeros((10)))
for i in range(0, len(model.get_SV())):
    sv=list(model.get_SV()[i].values())
    sv_modified=matrix(sv[0:10])
    w=w+alpha[i]*sv_modified
print(w)
#[[ 0.46679491  0.32202139  0.24630719 -0.19742437  0.56739051  0.00984055  -0.77086993  0.09004174 -0.01530599  0.18425151]]
#   BCL2L10     ZAR1L       C3orf56    BTG4         TUBB8       SH2D1B      C9orf116     TMEM132B   CA4          FAM19A4
#7,5,1
#7,5 accuracy:0.919214
#7,5,1 accuracy:0.947598