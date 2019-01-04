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
train_total=shuffledata(train_data,train_label,train_size,10)
train_data=train_total[:,index]
train_label=train_total[:,10]
#######################################################################################################
def selectdata( train_data,train_label,label,label_num ):#选取特定标签的数据
    specificlist = np.zeros((label_num[label],10))
    j=0
    for i in range(0,train_size):
        if train_label[i]==label:
            specificlist[j,:]=train_data[i,:]
            j=j+1
    return specificlist

data1=selectdata(train_data,train_label,0,label_num)
data2=selectdata(train_data,train_label,1,label_num)
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
#max_distinguish=-1
#for i in range(0,10):
#    for j in range(i,10):
#        for k in range(j,10):
#            temp=J_one_decide(data1,data2,[i,j,k])
#            if temp>max_distinguish:
#                max_distinguish=temp
#                print(i,end=" ")
#                print(j,end=" ")
#                print(k)
#######################################################################################################
def branch_and_bound_choosethree(data1,data2,remainfeatures,ignorefeatures):#分支定界算法
    global max_distinguish
    global finalfeature
    remainnum=len(remainfeatures)
    if remainnum==3:
        temp=J_one_decide(data1, data2, remainfeatures)
        if temp>max_distinguish:
            max_distinguish=temp;
            finalfeature=remainfeatures
        return
    if J_one_decide(data1, data2, remainfeatures)<max_distinguish:
        return
    else:
        J_one=[]
        tempfeatures_minus=[]
        for i in range(0,remainnum):
            if remainfeatures[i] in ignorefeatures:
                continue
            else:
                tempfeatures_minus=tempfeatures_minus+[remainfeatures[i]]
                J_one=J_one+[J_one_decide(data1, data2, np.delete(remainfeatures,i))]
        if len(J_one)<3:
            return
        J_one=np.array(J_one)
        min_1=np.where(J_one==np.min(J_one))
        min1=tempfeatures_minus[list(min_1)[0][0]]
        min_1=list(min_1)[0][0]
        J_one[min_1]=J_one[min_1]+10000

        min_2=np.where(J_one==np.min(J_one))
        min2=tempfeatures_minus[list(min_2)[0][0]]
        min_2=list(min_2)[0][0]
        J_one[min_2]=J_one[min_2]+10000

        min_3=np.where(J_one==np.min(J_one))
        min3=tempfeatures_minus[list(min_3)[0][0]]
        min_3=list(min_3)[0][0]
        temp=J_one_decide(data1, data2, [min1,min2,min3])
        if temp>max_distinguish:
            max_distinguish=temp;
            finalfeature=[min1,min2,min3]
        branch_and_bound_choosethree(data1,data2,np.delete(remainfeatures,min_1),ignorefeatures)
        branch_and_bound_choosethree(data1,data2,np.delete(remainfeatures,min_2),ignorefeatures+[min1])
        branch_and_bound_choosethree(data1,data2,np.delete(remainfeatures,min_3),ignorefeatures+[min1]+[min2])
#######################################################################################################
max_distinguish=-1
finalfeature=[0,1,2]
remainfeatures=np.array(range(0,10))
branch_and_bound_choosethree(data1,data2,remainfeatures,[])
list.sort(finalfeature)
print(max_distinguish)
print(finalfeature)
#[0, 1, 4]
#######################################################################################################
train_label=train_total[:,10]
train_data=train_total[:,[0,1,4]]
#######################################################################################################
model = svm_train(train_label,train_data,'-t 0')#线性核函数
print("Training data predict:")
predict1=svm_predict(train_label,train_data,model)
print("Test data predict:")
predict2=svm_predict(test_label,test_data,model)
#Training Accuracy = 0.953678
#Test Accuracy = 0.978022