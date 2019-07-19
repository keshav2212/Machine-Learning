import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
digits=load_digits()
k=dir(digits)
op=digit.images[0]
rg=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.1)
rg.fit(X_train,y_train)
y_predicted=rg.predict(X_test)
m=rg.score(X_test,y_test)
cm=confusion_matrix(y_test,y_predicted)
print(cm)    










