import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('salaries.csv')
inputs=df.drop('salary_more_then_100k',axis = 'columns')
target=df.salary_more_then_100k
print(inputs)
print(target)
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
print(inputs)
input_n=inputs.drop(['company','job','degree'],axis='columns')
print(input_n)
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(input_n,target)
print(model.score(input_n,target))
print(model.predict([[2,1,1]]))