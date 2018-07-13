# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np


fileName = "success_FR.txt"
f = open(fileName,"r")
ID = []
FR = []
fpi = []

f = open(fileName,"r+")
next(f)
for line in f:
    line = line.split()
    ID.append(float(line[0]))
    FR.append(float(line[1]))    

FRmax = max(FR) #max Frequency Ratio
fp_ind = [(100*x)/FRmax for x in FR] #flood probablity index (%)
fp_ind_sort = np.sort(fp_ind)        #flood probablity index sorted

T = len(fp_ind_sort)
N = 0
for i in fp_ind_sort:
    if i <=50:
        N += 1
P = T-N
print("Events: {}".format(T))
print("Positives: {}".format(P))
print("Negatives: {}".format(N))
        
x = np.linspace(min(fp_ind),max(fp_ind), 100)
roc_x = []
roc_y = []

TP = 0
FN = 0

for (i,j) in enumerate(x):
    for i in range(0,len(fp_ind_sort)):
        if (fp_ind_sort[i] < j) and (j<=50):
            FN += 1
        elif (fp_ind_sort[i] < j) and (j>50):
            TP += 1
    roc_x.append(FN/float(N))
    roc_y.append(TP/len(fp_ind_sort))
    FN = 0
    TP = 0
    
plt.plot(x,roc_y)
plt.show()