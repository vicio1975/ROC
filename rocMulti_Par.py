# -*- coding: utf-8 -*-
"""
created by: Vincenzo Sammartano
email: v.sammartano@gmail.com
"""
##IMPORT LIBS
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp

###FUNCTIONS section
def Figures(nf,X,Y):
    """
    Creation of figure series
    """
    font = {'family': 'Times New Roman',
            'color':  'black',
            'weight': 'normal',
            'size': 18,
            }
    
    plt.rcParams["font.family"] = "Times New Roman"
    fign = "figure"+str(nf)
    fig = plt.figure(fign)
    axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
    Cols = X[0][:,:].shape[1]
    for i in range(Cols):
        axes1.plot(X[0][:,i],Y[0][:,i],'k',label="success",linestyle="--",linewidth=2)
        axes1.plot(X[1][:,i],Y[1][:,i],'k',label="prediction",linestyle="-",linewidth=2)
        axes1.set_xlabel('False Positive Rate',fontdict=font)
        axes1.set_ylabel('True Positive Rate',fontdict=font)
##      x_ticks = np.linspace(0,1,5)
##      y_ticks = x_ticks
##      axes1.set_xticklabels(x_ticks, rotation=0, fontsize=16)
##      axes1.set_yticklabels(y_ticks, rotation=0, fontsize=16)
    figname = "roc_curve_"+str(nf)+".jpg"
    fig.savefig(figname,dpi=600)
    plt.show(fig)

def roc(ff, clas):
    """
    This is for ROC coefficient estimation
    """
    data = np.loadtxt(ff, delimiter="\t") #data array
    Nrow = data.shape[0]
    Ncol = data.shape[1]
    NcolOut = Ncol-1

    roc_x = np.ones((clas,NcolOut)) #output data
    roc_y = np.ones((clas,NcolOut)) #output data
  
    VF = data[:,0] #array of VF

    for i in range(1, Ncol):
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

    return XROC,YROC

def cores(clas, x, y, nf):
    """
    This is the core function of the code
    """
    fname = ["success"+str(nf)+".txt","prediction"+str(nf)+".txt"]
    for fi in fname:
        print(fi)
        Xroc,Yroc = roc(fi, clas)
        x.append(Xroc)
        y.append(Yroc)
        fi_out_Xroc = fi[:-4]+"_Xroc.txt"
        fi_out_Yroc = fi[:-4]+"_Yroc.txt"
        Time1 = time.time()
        print("Writing the file {}".format(fi_out_Xroc))
        np.savetxt(fi_out_Xroc, Xroc, delimiter="\t",fmt="%5.5f")
        print("Writing the file {}".format(fi_out_Yroc))
        np.savetxt(fi_out_Yroc, Yroc, delimiter="\t",fmt="%5.5f")
        print("Time per case {:5.3f} sec".format(time.time()-Time1))

    ###### Figure creation
    #Figures(nf,x,y)
    x = []
    y = []


###### END Function

###MAIN
if __name__ == "__main__":
    #startTime = time.time()
    print("# Parallel Computing #")
    clas = 1000 #Number of classes
    npr = []
    startTime = time.time()
    Nproc = 5  #mp.cpu_count()
    print("Number of processors: ", Nproc)
    X = []
    Y = []
    nfiles = list(range(1,5))
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(Nproc)
    # Step 2: `pool.apply` the `howmany_within_range()`
    pool.starmap(cores, [(clas, X, Y, nf) for nf in nfiles])
    pool.close()
    print("Total time {:5.3f} sec".format(time.time()-startTime))
