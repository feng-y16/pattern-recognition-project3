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
from draw import *
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
data=total[:,[4,6]]
label=total[:,10]
#######################################################################################################
model = svm_train(label,data,'-t 0')#线性核函数
print("Training data predict:")
predict=svm_predict(label,data,model)
#######################################################################################################
x_begin=min(data[:,0])#扫描分界面位置
x_end=max(data[:,0])
y_begin=min(data[:,1])-3
y_end=max(data[:,1])+1
pointnumber=1200
linex=np.linspace(x_begin,x_end,pointnumber)
liney=np.zeros(pointnumber)
linextest=np.ones(pointnumber)
tempdata=np.zeros((pointnumber,2))
tempdata[:,1]=np.linspace(y_begin,y_end,pointnumber)
def coutline():
    for i in range(0,pointnumber):
        tempdata[:,0]=linex[i]*linextest
        temp=svm_predict(np.zeros(pointnumber),tempdata,model,'-q')
        for j in range(1,pointnumber):
            if(temp[0][j]+temp[0][j-1]==1):
                break
        liney[i]=tempdata[j][1]
    return liney
coutline()
#######################################################################################################
#作图
plotall(data,label,np.array(predict[0]),None,None,None,train_size+test_size,0,False,linex,liney)
#######################################################################################################
data=total[:,[0,4,6]]#3features情形
label=total[:,10]
#######################################################################################################
model = svm_train(label,data,'-t 0')#线性核函数
print("Training data predict:")
predict=svm_predict(label,data,model)
#######################################################################################################
x_begin=min(data[:,0])#扫描分界面位置
x_end=max(data[:,0])
y_begin=min(data[:,1])-2
y_end=max(data[:,1])+1
z_begin=min(data[:,2])-1
z_end=max(data[:,2])+2
pointnumber=100
linex=np.linspace(x_begin,x_end,pointnumber)
liney=np.linspace(y_begin,y_end,pointnumber)
line=np.ones(pointnumber)
tempdata=np.zeros((pointnumber,3))
meshz=np.zeros((pointnumber,pointnumber))
tempdata[:,2]=np.linspace(z_begin-4.5,z_end+0.5,pointnumber)
def coutline():
    for i in range(0,pointnumber):
        for j in range(0,pointnumber):
            tempdata[:,0]=linex[i]*line
            tempdata[:,1]=liney[j]*line
            temp=svm_predict(np.zeros(pointnumber),tempdata,model,'-q')
            for k in range(1,pointnumber):
                if(temp[0][k]+temp[0][k-1]==1):
                    break
            meshz[i][j]=tempdata[k][2]
    return meshz
#coutline()
#######################################################################################################
#作图
plot3d(data,label,np.array(predict[0]),None,None,None,train_size+test_size,0,False,linex,liney,meshz)
