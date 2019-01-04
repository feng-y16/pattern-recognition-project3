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
import copy

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
datatemp=np.zeros((train_size+test_size,10))
labeltemp=np.zeros(train_size+test_size)
datatemp[range(0,train_size),:]=train_data
datatemp[range(train_size,train_size+test_size),:]=test_data
labeltemp[range(0,train_size)]=train_label
labeltemp[range(train_size,train_size+test_size)]=test_label
total=shuffledata(datatemp,labeltemp,train_size+test_size,10)
data=total[:,index]
label=total[:,10]
#######################################################################################################
def selectdata( train_data,train_label,label,label_num ):#选取特定标签的数据
    specificlist = np.zeros((label_num[label],10))
    j=0
    for i in range(0,train_size+test_size):
        if train_label[i]==label:
            specificlist[j,:]=train_data[i,:]
            j=j+1
    return specificlist
label_num2=[0,0]
for i in range(0,test_size):
    if test_label[i]==0:
        label_num2[0]=label_num2[0]+1
    else:
        label_num2[1]=label_num2[1]+1
label_num[0]=label_num2[0]+label_num[0]
label_num[1]=label_num2[1]+label_num[1]
data1=selectdata(data,label,0,label_num)
data2=selectdata(data,label,1,label_num)
#######################################################################################################
def J_one_decide(data1, data2, remainfeatures):#计算判据的值
    remainnum=len(remainfeatures)
    deletenum=10-remainnum
    data1processed=np.zeros((len(data1),remainnum))
    data2processed=np.zeros((len(data2),remainnum))
    j=0
    for i in range(0,10):
        if i in remainfeatures:
            data1processed[:,j]=data1[:,i]
            data2processed[:,j]=data2[:,i]
            j=j+1
        else:
           continue
    data1processed_mean=matrix(np.mean(data1processed,axis=0))
    data2processed_mean=matrix(np.mean(data2processed,axis=0))
    S1=matrix(np.zeros((remainnum,remainnum)))
    S2=matrix(np.zeros((remainnum,remainnum)))
    for i in range(0,len(data1)):
        S1=S1+matrix(data1processed[i,:]).T*data1processed_mean
    for i in range(0,len(data2)):
        S2=S2+matrix(data2processed[i,:]).T*data2processed_mean
    Sw=S1+S2
    Sb=matrix(data1processed_mean-data2processed_mean).T*matrix(data1processed_mean-data2processed_mean)
    return np.trace(Sw+Sb)
max_distinguish=-1
for i in range(0,10):
    for j in range(i,10):
        for k in range(j,10):
            temp=J_one_decide(data1,data2,[i,j,k])
            if temp>max_distinguish:
                max_distinguish=temp
                print(i,end=" ")
                print(j,end=" ")
                print(k)
                print(temp)
