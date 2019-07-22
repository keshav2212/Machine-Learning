import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris=load_iris()
#print(df)
print(dir(iris))
print(iris.feature_names)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
df['target']=iris.target
print(df)
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
print(df)
df0=df[:50]
df1=df[50:100]
df2=df[100:]
#print(df0)
#print(df1)
#print(df2)
#plt.xlabel('sepal length')
#plt.ylabel('sepal width')
#plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],marker="*",color="green")
#plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],marker="*",color="red")
#plt.xlabel('petal length')
#plt.ylabel('petal width')
#plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color="green")
#plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color="red")
X=df.drop(['target','flower_name'],axis='columns')
y=df.target
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.svm import SVC
#model=SVC(C=1)
#model.fit(X_train,y_train)
#print(model.score(X_test,y_test))
model_linear_kernal=SVC(kernel='linear')
model_linear_kernal.fit(X_train,y_train)
print(model_linear_kernal.score(X_test,y_test))








