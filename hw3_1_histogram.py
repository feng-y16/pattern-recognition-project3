import numpy as np
import math
from numpy import *
from collections import OrderedDict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#r=np.load("10genes_withoutnorm.npz")
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
r1=range(0,5)
r2=range(5,10)
train_data1=train_data[:,r1]
train_data2=train_data[:,r2]
test_data1=test_data[:,r1]
test_data2=test_data[:,r2]
#bins = np.arange(-2, 4, 0.5) 
colors1=['darkorange','orange','gold','yellow','darkkhaki']
colors2=['darkslategray','teal','cyan','deepskyblue','lightblue']
#colors3=['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue']
#colors4=['black','dimgray','dimgrey','grey','gray','darkgrey','darkgray','silver','lightgray','lightgrey']
#colors5=list(reversed(colors2))
plt.hist(train_data1,  label = 'train_data', color = colors1, alpha = 0.7) 
plt.hist(test_data1,  label = 'test_data', color = colors2,alpha = 0.6) 
plt.title( 'histogram(features 1~5)')
plt.xlabel( 'range')
plt.ylabel( 'numbers occured') 
plt.tick_params(top= 'off', right= 'off') 
plt.legend() 
plt.show() 

plt.hist(train_data2,  label = 'train_data', color = colors1, alpha = 0.7) 
plt.hist(test_data2,  label = 'test_data', color = colors2,alpha = 0.6) 
plt.title( 'histogram(features 6~10)')
plt.xlabel( 'range')
plt.ylabel( 'numbers occured') 
plt.tick_params(top= 'off', right= 'off') 
plt.legend() 
plt.show() 