import matplotlib.pyplot as plt
import numpy as np

acc = open('pac_acc.txt','r')
f1 = open('pac_f.txt','r')
pre = open('pac_precision.txt','r')
rec = open('pac_recall.txt','r')
y = list(map(lambda x : float(x.strip()),acc.readlines()))
y_f = list(map(lambda x : float(x.strip()),f1.readlines()))
y_p = list(map(lambda x : float(x.strip()),pre.readlines()))
y_r = list(map(lambda x : float(x.strip()),rec.readlines()))

figure, axis = plt.subplots(2, 2)
axis[0, 0].plot(y)
axis[0, 0].set_title("accuracy")
  
axis[0, 1].plot(y_f)
axis[0, 1].set_title("f1 score")
  
axis[1, 0].plot(y_p)
axis[1, 0].set_title("precision")
  
axis[1, 1].plot(y_r)
axis[1, 1].set_title("recall")
plt.xlabel("Batch #")

plt.legend()
plt.show()
