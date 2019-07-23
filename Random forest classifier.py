import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digit=load_digits()
#for i in range(4):
 #   plt.matshow(digit.images[i])
df=pd.DataFrame(digit.data)
df['target']=digit.target
X=df.drop(df.target,axis="columns")
y=df.target
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))