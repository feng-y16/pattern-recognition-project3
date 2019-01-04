import numpy as np
import random

def shuffledata (data, label, size, feature):
    random.seed(1)
    label_num=[0,0]
    total=np.zeros((size,feature+1))
    i=range(0,feature)
    total[:,i]=data
    total[:,feature]=label
    data_total=np.zeros((size,feature+1))
    i=range(0,size)
    i=np.array(i)
    random.shuffle(i)
    for j in range(0,size):
        data_total[j,:]=total[i[j],:]
        if data_total[j,feature]==0:
            label_num[0]=label_num[0]+1
        else:
            label_num[1]=label_num[1]+1
    return data_total
