import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import time
train = pd.read_csv('train.csv')
train = train.sample(frac=1.0, axis=0)  # shuffle the data
train.fillna(-999, inplace=True)

test = pd.read_csv('test.csv')
test.fillna(-999, inplace=True)

train_y = train.label.values
train_X = train.drop('label', axis=1).values

test_y = test.label.values
test_X = test.drop('label', axis=1).values

del train, test
print (train_X.shape, test_X.shape)

t1 = time.time()
clf = RandomForestClassifier(n_estimators = 100,max_depth=15,)
#clf = DecisionTreeClassifier(random_state=0)

clf.fit(train_X,train_y)
t2 = time.time()
print("training used:"+str(t2-t1))
y_test_pred = clf.predict(test_X)
t3 = time.time()
print("predict uesd:"+str(t3-t2))
f1 = f1_score(test_y, y_test_pred)
print(f1)