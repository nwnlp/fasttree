
from sklearn.metrics import *
truth = []
pred = []
for line in open('output.txt','r').readlines():
    v1, v2 = line.strip().split(' ')
    truth.append(int(v1))
    pred.append(int(v2))

f1 = f1_score(truth, pred)
recall = recall_score(truth,pred)
print(f1, recall)