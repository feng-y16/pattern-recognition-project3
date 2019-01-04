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
#min_entrophy=10000
#for i in range(0,10):
#    for j in range(i,10):
#        temp=H_two_decide(data1,data2,[i,j])
#        if temp<min_entrophy:
#            min_entrophy=temp
#            print(i,end=" ")
#            print(j)
#            print(temp)
#######################################################################################################
def branch_and_bound_choosetwo(data1,data2,remainfeatures,ignorefeatures):#分支定界算法
    global min_entrophy
    global finalfeature
    remainnum=len(remainfeatures)
    H_two=[]
    tempfeatures_minus=[]
    if remainnum==2:
        temp=H_two_decide(data1, data2, remainfeatures)
        if temp<min_entrophy:
            min_entrophy=temp;
            finalfeature=remainfeatures
        return
    for i in range(0,remainnum):
        if remainfeatures[i] in ignorefeatures:
            continue
        else:
            tempfeatures_minus=tempfeatures_minus+[remainfeatures[i]]
            H_two=H_two+[H_two_decide(data1, data2, np.delete(remainfeatures,i))]
    if len(H_two)<2:
        return;
    H_two=np.array(H_two)
    max_1=np.where(H_two==np.max(H_two))
    max1=tempfeatures_minus[list(max_1)[0][0]]
    max_1=list(max_1)[0][0]
    H_two[max_1]=H_two[max_1]-10000

    max_2=np.where(H_two==np.max(H_two))
    max2=tempfeatures_minus[list(max_2)[0][0]]
    max_2=list(max_2)[0][0]
    temp=H_two_decide(data1, data2, [max1,max2])
    if temp<min_entrophy:
        min_entrophy=temp
        finalfeature=[max1,max2]
    branch_and_bound_choosetwo(data1,data2,np.delete(remainfeatures,max_1),ignorefeatures+[max2])
    branch_and_bound_choosetwo(data1,data2,np.delete(remainfeatures,max_2),ignorefeatures)
#######################################################################################################
min_entrophy=10000
finalfeature=[0,1]
remainfeatures=np.array(range(0,10))
branch_and_bound_choosetwo(data1,data2,remainfeatures,[])
list.sort(list(finalfeature))
print(min_entrophy)
print(finalfeature)
#[5, 8]
#######################################################################################################
train_label=train_total[:,10]
train_data=train_total[:,[5,8]]
#######################################################################################################
model = svm_train(train_label,train_data,'-t 0')#线性核函数
print("Training data predict:")
predict1=svm_predict(train_label,train_data,model)
print("Test data predict:")
predict2=svm_predict(test_label,test_data,model)
#Training Accuracy = 0.822888
#Test Accuracy = 0.824176