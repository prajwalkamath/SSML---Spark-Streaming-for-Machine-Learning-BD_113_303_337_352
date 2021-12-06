import time
import sys
import json
import requests
import pickle
import numpy as np

sacc = open("sgd_acc.txt","r")
sf = open("sgd_f.txt","r")
sre = open("sgd_recall.txt","r")
spr = open("sgd_precision.txt","r")

macc = open("nb_acc.txt","r")
mf = open("nb_f.txt","r")
mre = open("nb_recall.txt","r")
mpr = open("nb_precision.txt","r")

pacc = open("pac_acc.txt","r")
pf = open("pac_f.txt","r")
pre = open("pac_recall.txt","r")
ppr = open("pac_precision.txt","r")

fp = open("metricresults.txt",'w+')
sLinesa = sacc.readlines()
sLinesf = sf.readlines()
sLinesre = sre.readlines()
sLinesp = spr.readlines()

mLinesa = macc.readlines()
mLinesf = mf.readlines()
mLinesre = mre.readlines()
mLinesp = mpr.readlines()

pLinesa = pacc.readlines()
pLinesf = pf.readlines()
pLinesre = pre.readlines()
pLinesp = ppr.readlines()
count=0
sum1=0
for linea in sLinesa:
	count+=1
	sum1+=float(linea)
fp.write(f"Accuracy of SGD Classifier:{sum1/count}\n")
sum2=0
for linef in sLinesf:
	sum2+=float(linef)
fp.write(f"F1 Score of SGD Classifier:{sum2/count}\n")
sum3=0
for linere in sLinesre:
	sum3+=float(linere)
fp.write(f"Recall of SGD Classifier:{sum3/count}\n")
sum4=0
for linep in sLinesp:
	sum4+=float(linep)
fp.write(f"Precision of SGD Classifier:{sum4/count}\n")

count=0
sum1=0
for linea in pLinesa:
	count+=1
	sum1+=float(linea)
fp.write(f"Accuracy of PAC Classifier:{sum1/count}\n")
sum2=0
for linef in pLinesf:
	sum2+=float(linef)
fp.write(f"F1 Score of PAC Classifier:{sum2/count}\n")
sum3=0
for linere in pLinesre:
	sum3+=float(linere)
fp.write(f"Recall of PAC Classifier:{sum3/count}\n")
sum4=0
for linep in pLinesp:
	sum4+=float(linep)
fp.write(f"Precision of PAC Classifier:{sum4/count}\n")

count=0
sum1=0
for linea in mLinesa:
	count+=1
	sum1+=float(linea)
fp.write(f"Accuracy of Multinomial Classifier:{sum1/count}\n")
sum2=0
for linef in mLinesf:
	sum2+=float(linef)
fp.write(f"F1 Score of Multinomial Classifier:{sum2/count}\n")
sum3=0
for linere in mLinesre:
	sum3+=float(linere)
fp.write(f"Recall of Multinomial Classifier:{sum3/count}\n")
sum4=0
for linep in mLinesp:
	sum4+=float(linep)
fp.write(f"Precision of Multinomial Classifier:{sum4/count}\n")

	

