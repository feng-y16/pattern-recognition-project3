import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import *
from sklearn import preprocessing
offset=1
#######################################################################################################
train_original_data=pd.read_csv("E:\\train_data_E3E5_10genes.txt",sep=' ')#处理训练集
train_size=len(train_original_data["BCL2L10"])
train_data=np.zeros((train_size,10))
train_label=np.zeros(train_size)
label_num=[0,0]
label_num_total=[0,0]
for i in range(0,train_size):
    train_data[i,0]=math.log(train_original_data.ix[i]["BCL2L10"]+offset)
    train_data[i,1]=math.log(train_original_data.ix[i]["ZAR1L"]+offset)
    train_data[i,2]=math.log(train_original_data.ix[i]["C3orf56"]+offset)
    train_data[i,3]=math.log(train_original_data.ix[i]["BTG4"]+offset)
    train_data[i,4]=math.log(train_original_data.ix[i]["TUBB8"]+offset)
    train_data[i,5]=math.log(train_original_data.ix[i]["SH2D1B"]+offset)
    train_data[i,6]=math.log(train_original_data.ix[i]["C9orf116"]+offset)
    train_data[i,7]=math.log(train_original_data.ix[i]["TMEM132B"]+offset)
    train_data[i,8]=math.log(train_original_data.ix[i]["CA4"]+offset)
    train_data[i,9]=math.log(train_original_data.ix[i]["FAM19A4"]+offset)
    #train_data[i,0]=math.log(math.log(train_original_data.ix[i]["BCL2L10"]+offset)+offset)
    #train_data[i,1]=math.log(math.log(train_original_data.ix[i]["ZAR1L"]+offset)+offset)
    #train_data[i,2]=math.log(math.log(train_original_data.ix[i]["C3orf56"]+offset)+offset)
    #train_data[i,3]=math.log(math.log(train_original_data.ix[i]["BTG4"]+offset)+offset)
    #train_data[i,4]=math.log(math.log(train_original_data.ix[i]["TUBB8"]+offset)+offset)
    #train_data[i,5]=math.log(math.log(train_original_data.ix[i]["SH2D1B"]+offset)+offset)
    #train_data[i,6]=math.log(math.log(train_original_data.ix[i]["C9orf116"]+offset)+offset)
    #train_data[i,7]=math.log(math.log(train_original_data.ix[i]["TMEM132B"]+offset)+offset)
    #train_data[i,8]=math.log(math.log(train_original_data.ix[i]["CA4"]+offset)+offset)
    #train_data[i,9]=math.log(math.log(train_original_data.ix[i]["FAM19A4"]+offset)+offset)
    if train_original_data.ix[i]["label"]=="E3":
        train_label[i]=0
        label_num[0]=label_num[0]+1
        label_num_total[0]=label_num_total[0]+1
    else:
        train_label[i]=1
        label_num[1]=label_num[1]+1
        label_num_total[1]=label_num_total[1]+1
#######################################################################################################
test_original_data=pd.read_csv("E:\\test_data_E3E5_10genes.txt",sep=' ')#处理测试集
test_size=len(test_original_data["BCL2L10"])
test_data=np.zeros((test_size,10))
test_label=np.zeros(test_size)
for i in range(0,test_size):
    test_data[i,0]=math.log(test_original_data.ix[i]["BCL2L10"]+offset)
    test_data[i,1]=math.log(test_original_data.ix[i]["ZAR1L"]+offset)
    test_data[i,2]=math.log(test_original_data.ix[i]["C3orf56"]+offset)
    test_data[i,3]=math.log(test_original_data.ix[i]["BTG4"]+offset)
    test_data[i,4]=math.log(test_original_data.ix[i]["TUBB8"]+offset)
    test_data[i,5]=math.log(test_original_data.ix[i]["SH2D1B"]+offset)
    test_data[i,6]=math.log(test_original_data.ix[i]["C9orf116"]+offset)
    test_data[i,7]=math.log(test_original_data.ix[i]["TMEM132B"]+offset)
    test_data[i,8]=math.log(test_original_data.ix[i]["CA4"]+offset)
    test_data[i,9]=math.log(test_original_data.ix[i]["FAM19A4"]+offset)
    #test_data[i,0]=math.log(math.log(test_original_data.ix[i]["BCL2L10"]+offset)+offset)
    #test_data[i,1]=math.log(math.log(test_original_data.ix[i]["ZAR1L"]+offset)+offset)
    #test_data[i,2]=math.log(math.log(test_original_data.ix[i]["C3orf56"]+offset)+offset)
    #test_data[i,3]=math.log(math.log(test_original_data.ix[i]["BTG4"]+offset)+offset)
    #test_data[i,4]=math.log(math.log(test_original_data.ix[i]["TUBB8"]+offset)+offset)
    #test_data[i,5]=math.log(math.log(test_original_data.ix[i]["SH2D1B"]+offset)+offset)
    #test_data[i,6]=math.log(math.log(test_original_data.ix[i]["C9orf116"]+offset)+offset)
    #test_data[i,7]=math.log(math.log(test_original_data.ix[i]["TMEM132B"]+offset)+offset)
    #test_data[i,8]=math.log(math.log(test_original_data.ix[i]["CA4"]+offset)+offset)
    #test_data[i,9]=math.log(math.log(test_original_data.ix[i]["FAM19A4"]+offset)+offset)
    if test_original_data.ix[i]["label"]=="E3":
        test_label[i]=0
        label_num_total[0]=label_num_total[0]+1
    else:
        test_label[i]=1
        label_num_total[1]=label_num_total[1]+1
#######################################################################################################
np.savez("10genes_withoutnorm.npz",train_data,train_label,test_data,test_label,label_num,train_size,test_size,label_num_total)
train_range=range(0,train_size)
test_range=range(train_size,train_size+test_size)
data=np.zeros((train_size+test_size,10))
data[train_range,:]=train_data
data[test_range,:]=test_data
for i in range(0,10):
    data[:,i]=sklearn.preprocessing.scale(data[:,i])
train_data=data[train_range,:]
test_data=data[test_range,:]
np.savez("10genes.npz",train_data,train_label,test_data,test_label,label_num,train_size,test_size,label_num_total)
