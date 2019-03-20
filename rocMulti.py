# -*- coding: utf-8 -*-
"""

"""
##IMPORT LIBS
import matplotlib.pyplot as plt
import numpy as np
import time
clas = 1000
startTime = time.time()
###FUNCTIONS
def roc(ff, clas):
    data = np.loadtxt(ff, delimiter="\t") #data array
    Nrow = data.shape[0]
    Ncol = data.shape[1]
    NcolOut = Ncol-1

    roc_x = np.ones((clas,NcolOut)) #output data
    roc_y = np.ones((clas,NcolOut)) #output data
  
    VF = data[:,0] #array of VF

    for i in range(1,Ncol):
        if (str(ff[-5]) == "4"):
            FRmin = abs(min(data[:,i]))
            data[:,i] = data[:,i] + FRmin
        else:
            FRmax = max(data[:,i]) #max Frequency Ratio
            data[:,i] = 100*data[:,i]/FRmax #flood probablity index (%) Flood Events
    TP  = 0
    FN  = 0
    TN  = 0
    FP  = 0
    for i in range(1, Ncol):
        x = np.linspace(0, max(data[:,i]), clas)
        for (ix,frx) in enumerate(x):
            for j in range(Nrow):
                if    (data[j,i] <= frx) and (VF[j] == 0):
                    TN += 1
                elif  (data[j,i] <= frx) and (VF[j] == 1):     
                    FN += 1
                elif    (data[j,i] > frx ) and (VF[j] == 1):
                    TP += 1
                elif  (data[j,i] > frx ) and (VF[j] == 0):     
                    FP += 1
            
            roc_x[ix, i-1] = (FP/(FP+TN))
            roc_y[ix, i-1] = (TP/(TP+FN))
            #print(roc_x[ix,i],roc_y[ix,i])
            TP = 0
            FN = 0
            TN = 0
            FP = 0
    
    if (str(ff[-5]) == "2"):
        XROC = roc_y
        YROC = roc_x
    else:    
        XROC = roc_x
        YROC = roc_y

    return XROC,YROC #,area
###### END


###MAIN
X = []
Y = []
As = []
Ap = []


fname = []

font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 18,
        }    ##Data to process

lst = ['-', "--"]
plt.rcParams["font.family"] = "Times New Roman"
X = []
Y = []
for nf in range(1,5):
    fname = ["success"+str(nf)+".txt","prediction"+str(nf)+".txt"]
    
    for fi in fname:
        print(fi)
        Xroc,Yroc = roc(fi,clas)
        X.append(Xroc)
        Y.append(Yroc)
    fign = "figure"+str(nf)
    fig = plt.figure(fign)
    axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
        
    for i in range(Xroc.shape[1]):
        axes1.plot(X[0][:,i],Y[0][:,i],'k',label="success",linestyle="--",linewidth=2)
        axes1.plot(X[1][:,i],Y[1][:,i],'k',label="prediction",linestyle="-",linewidth=2)
        lx = axes1.set_xlabel('False Positive Rate',fontdict=font)
        ly = axes1.set_ylabel('True Positive Rate',fontdict=font)
##        x_ticks = np.linspace(0,1,5)
##        y_ticks = x_ticks
##        axes1.set_xticklabels(x_ticks, rotation=0, fontsize=16)
##        axes1.set_yticklabels(y_ticks, rotation=0, fontsize=16)
    figname = "roc_curve_"+str(nf)+".jpg"
    fig.savefig(figname,dpi=600)
    plt.show(fig)
    X = []
    Y = []
print("Total time {}".format(time.time()-startTime))
