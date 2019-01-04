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
def H_two_decide(data1, data2, remainfeatures):#计算判据的值
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
    P1=len(data1processed)/(len(data1processed)+len(data2processed))
    P2=len(data2processed)/(len(data1processed)+len(data2processed))
    data1processed_cov=matrix(np.cov(data1processed.T))*(len(data1processed)-1)/len(data1processed)
    data2processed_cov=matrix(np.cov(data2processed.T))*(len(data2processed)-1)/len(data2processed)
    try:
        inverse1=data1processed_cov.I
    except:
        data1processed_cov=data1processed_cov+0.01*np.identity(len(remainfeatures))
        inverse1=(data1processed_cov).I
    try:
        inverse2=data2processed_cov.I
    except:
        data2processed_cov=data2processed_cov+0.01*np.identity(len(remainfeatures))
        inverse2=(data2processed_cov).I
    b1=-0.5*math.log(abs(np.linalg.det(data1processed_cov)))+math.log(P1)-0.5*len(remainfeatures)*math.log(2*math.pi)
    b2=-0.5*math.log(abs(np.linalg.det(data2processed_cov)))+math.log(P2)-0.5*len(remainfeatures)*math.log(2*math.pi)
    H_two=0
    for i in range(0,len(data1processed)):
        x=matrix(data1processed[i,:])
        exp1=-0.5*np.dot(np.dot((x-data1processed_mean),inverse1),(x-data1processed_mean).T)+b1
        exp2=-0.5*np.dot(np.dot((x-data2processed_mean),inverse2),(x-data2processed_mean).T)+b2
        k=1/(math.exp(exp1)+math.exp(exp2))
        H_two=H_two+2*(1-k*math.exp(2*exp1)*k-k*math.exp(2*exp2)*k)
    for i in range(0,len(data2processed)):
        x=matrix(data2processed[i,:])
        exp1=-0.5*np.dot(np.dot((x-data1processed_mean),inverse1),(x-data1processed_mean).T)+b1
        exp2=-0.5*np.dot(np.dot((x-data2processed_mean),inverse2),(x-data2processed_mean).T)+b2
        k=1/(math.exp(exp1)+math.exp(exp2))
        H_two=H_two+2*(1-k*math.exp(2*exp1)*k-k*math.exp(2*exp2)*k)
    return H_two
min_entrophy=10000
for i in range(0,10):
    for j in range(i,10):
        for k in range(j,10):
            temp=H_two_decide(data1,data2,[i,j,k])
            if temp<min_entrophy:
                min_entrophy=temp
                print(i,end=" ")
                print(j,end=" ")
                print(k)
                print(temp)
