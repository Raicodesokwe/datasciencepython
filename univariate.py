import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import quandl
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas.plotting as pp

boston_dataset=load_boston()
boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target
X2 = boston[['RM', 'LSTAT']]
Y = boston['MEDV']
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y,
                                                      test_size=0.3,
                                                      random_state=1)

model2 = LinearRegression()
model2.fit(X2_train,Y_train)
Y_test_predicted2 = model2.predict(X2_test)
print(Y_test_predicted2)
print(mean_squared_error(Y_test,Y_test_predicted2))
iriswarriris=pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
print(iriswarriris.head())
print(iriswarriris.describe())
print(iriswarriris.groupby('species').size())
print(iriswarriris['species'].value_counts())
print(iriswarriris.drop('id',axis=1,inplace=True))


pp.scatter_matrix(iriswarriris)
sns.catplot(x='sepal_wd',y='sepal_len',hue='species',data=iriswarriris)