# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

fileName = "success_FR.txt"
f = open(fileName,"r")
ID = []
FR = []
ll = []

f = open(fileName,"r+")
next(f)
for line in f:
    line = line.split()
    ID.append(float(line[0]))
    FR.append(float(line[1]))    

fpi = []

FRmax = max(FR); #max Frequency Ratio
print(FRmax)

fpi = [(100*x)/FRmax for x in FR] #flood probablity index (%)

