import numpy as np
import math
from numpy import *
from collections import OrderedDict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plotline(data,label,decide,size,feature):
    plt.scatter(x=data[:,0],y=data[:,1],s=10,c=label)
    plt.show()
    return 0
#######################################################################################################
def plotall(train_data,train_label,train_decide,test_data,test_label,test_decide,train_size,test_size,test_exist=False,linex=None,liney=None):
    colors = ['b','g','r','orange']#作图(2D)
    labelpool=['E3_right','E3_wrong','E5_right','E5_wrong']
    train_decide=train_decide.astype(int)
    for i in range(0,train_size):
        if train_label[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='o',label=labelpool[train_decide[i]])
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    if(test_exist):
        test_decide=test_decide.astype(int)
        for i in range(0,test_size):
            if test_label[i]==0:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='o',label=labelpool[train_decide[i]])
            else:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    plt.plot(linex,liney,color="red")
    plt.xlabel('normalized_log(TUBB8+1)')
    plt.ylabel('normalized_log(C9orf116+1)')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0]
    for j in range(0,4):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,4):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys,loc = 'upper left')
    plt.title("SVM_2features")
    #plt.savefig("H2_2features.png")
    plt.show()
    return 0
#######################################################################################################
def plot3d(train_data,train_label,train_decide,test_data,test_label,test_decide,train_size,test_size,test_exist=False,linex=None,liney=None,meshz=None):
    colors = ['b','g','r','orange']#作图(3D)
    labelpool=['E3_right','E3_wrong','E5_right','E5_wrong']
    train_decide=train_decide.astype(int)
    fig = plt.figure()
    ax = Axes3D(fig)
    z_begin=min(train_data[:,2])
    z_end=max(train_data[:,2])+0.2
    dz=z_end-z_begin
    for i in range(0,train_size):
        if train_label[i]==0:
            ax.scatter(xs=train_data[i,0],ys=train_data[i,1],zs=train_data[i,2],s=9,alpha=(z_end-train_data[i,2])/dz,c=colors[train_decide[i]],marker='o',label=labelpool[train_decide[i]])
        else:
            ax.scatter(xs=train_data[i,0],ys=train_data[i,1],zs=train_data[i,2],s=9,alpha=(z_end-train_data[i,2])/dz,c=colors[train_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    if(test_exist):
        test_decide=test_decide.astype(int)
        for i in range(0,test_size):
            if test_label[i]==0:
                ax.scatter(xs=test_data[i,0],ys=test_data[i,1],zs=train_data[i,2],s=9,c=colors[test_decide[i]],marker='o',label=labelpool[train_decide[i]])
            else:
                ax.scatter(xs=test_data[i,0],ys=test_data[i,1],zs=train_data[i,2],s=9,c=colors[test_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    x, y = np.meshgrid(linex, liney)

    #ax.plot_surface(x, y, meshz, rstride=1, cstride=1, cmap=plt.cm.jet,alpha=0.4)
    #ax.set_xlabel('normalized_log(BTG4+1)')
    #ax.set_ylabel('normalized_log(SH2D1B+1)')
    #ax.set_zlabel('normalized_log(CA4+1)')

    #ax.set_xlabel('normalized_log(BCL2L10+1)')
    #ax.set_ylabel('normalized_log(ZAR1L+1)')
    #ax.set_zlabel('normalized_log(TUBB8+1)')

    ax.set_xlabel('normalized_log(BCL2L10+1)')
    ax.set_ylabel('normalized_log(TUBB8+1)')
    ax.set_zlabel('normalized_log(C9orf116+1)')

    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0]
    for j in range(0,4):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,4):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys,loc = 'upper left')
    #plt.title("H2_3features")
    #plt.title("J1_3features")
    plt.title("SVM_3features")
    plt.show()
    return 0