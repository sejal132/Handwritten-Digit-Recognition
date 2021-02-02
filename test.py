import numpy as np 
import matplotlib.pyplot as pt
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv(r"D:\Python\Handwritten Digit Recognition_2\train.csv").to_numpy()
clf=DecisionTreeClassifier()
print(data)

#training dataset
xtrain=data[:21000,1:]
train_label=data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label=data[21000:,0]

p=clf.predict(xtest)
count=0
for i in range(0,21000):
    count+=1 if p[i]==actual_label[i] else 0
print("Accuracy is : {}".format((count/21000)*100))





