import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import quandl
from sklearn.datasets import load_boston,load_wine,load_breast_cancer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas.plotting as pp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix,precision_score,recall_score,f1_score,precision_recall_fscore_support
from sklearn.cluster import KMeans
from math import sqrt
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer
quandl.ApiConfig.api_key='_EmpxxFMfoJyjMuvso8h'


from sklearn.datasets import load_boston
boston_dataset = load_boston()
## build a DataFrame
boston = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
StandardScaler()
model=LinearRegression()

model.fit(X_train,Y_train)
model.intercept_
model.coef_
Y_test_predicted=model.predict(X_test)

iriswarriris=pd.read_csv('https://sololearn.com/uploads/files/iris.csv')

print(iriswarriris.describe())
print(iriswarriris.groupby('species').size())
print(iriswarriris['species'].value_counts())
print(iriswarriris.drop('id',axis=1,inplace=True))
print(iriswarriris.head())


pp.scatter_matrix(iriswarriris)
fig=sns.catplot(x='petal_wd',y='petal_len',hue='species',data=iriswarriris)
fig.set_axis_labels('Petal width in cm','petal length in cm')
sns.pairplot(iriswarriris)
plt.show()
x=iriswarriris[['petal_len','petal_wd']]
y=iriswarriris['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3,stratify=y)
knn=KNeighborsClassifier(n_neighbors=5)
print(knn.fit(x_train,y_train))
pred=knn.predict(x_test)
print(pred[10:12])
posibo=knn.predict_proba(x_test)
print(posibo[10:12])

print(knn.score(x_test,y_test))
print(accuracy_score(y_test,pred))
print((pred==y_test.values).sum()/y_test.size)
print(y_test.size)
print(confusion_matrix(y_test,pred,labels=['iris-setosa','iris-virginica','iris-versicolor']))
plot_confusion_matrix(knn,x_test,y_test,cmap=plt.cm.Blues)
plt.show()
y_test=np.array(['cat','mouse','cat','mouse','cat'])
y_pred=np.array(['cat','mouse','cat','mouse','mouse'])
print(confusion_matrix(y_test,y_pred,labels=['cat','mouse']))
knn2=KNeighborsClassifier(n_neighbors=3)
knn_cv=cross_val_score(knn2,x,y,cv=5)
print(knn_cv)
print(knn_cv.mean())
knn=KNeighborsClassifier()
parambulate={'n_neighbors':np.arange(1,10)}
knn_gsv=GridSearchCV(knn,parambulate,cv=5)
knn_gsv.fit(x,y)
print(knn_gsv.best_params_)
print(knn_gsv.best_score_)
print()
knn_final=KNeighborsClassifier(knn_gsv.best_params_['n_neighbors'])
knn_final.fit(x,y)
knn_final.predict(x)
print(knn_final.score(x,y))
print(knn_final.predict([[3.76,1.20]]))

new_data=np.array([[5.03,1.2],[3.76,1.2],[1.58,1.2]])
print(knn_final.predict(new_data))
print(knn_final.predict_proba(new_data))
x1=np.array([1,-1])
x2=np.array([4,3])
print(np.sqrt(((x1-x2)**2).sum()))

wine=load_wine()

df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['group']=wine.target
df['alcohoo']=df['alcohol']==14.23
print(df.head())
arr=df[['alcohol','malic_acid']].values
print(arr[:,1]>4.0)
pp.scatter_matrix(df.iloc[:,[0,5]])
plt.show()
inertias=[]
mapping={}
K=range(1,10)
for k in K:
    kmeansklasta=KMeans(n_clusters=k).fit(df)
    kmeansklasta.fit(df)

    inertias.append(kmeansklasta.inertia_)
    mapping[k]=kmeansklasta.inertia_

plt.plot(K,inertias,'bx-')
plt.xlabel('Values fi K')
plt.ylabel('Inertia')
plt.title('Finding elbow point through inertia')
plt.show()
print(df.head())
print(df.loc[0:2,:])
print(df.loc[0,:])
print(df[df.alcohol==14.23])

X=df[['alcohol','total_phenols']]
scale=StandardScaler()
X_scaled=scale.fit_transform(X)

plt.scatter(X_scaled[:,0],X_scaled[:,1],marker='.',label='Scaled')
plt.scatter(X.iloc[:,[0]],X.iloc[:,[1]],marker='*',label='original data')
plt.legend()
plt.show()

senke=KMeans()
vixuo=KElbowVisualizer(senke,k=(1,10))
vixuo.fit(df)
vixuo.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(X_scaled)
y_prede=kmeans.predict(X_scaled)
print(y_prede)
print(kmeans.cluster_centers_)

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y_prede)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c=[0,1,2],marker='*',s=250,edgecolors='k')
plt.xlabel('alcohoo')
plt.ylabel('total phenols')
plt.title('centroids(3 clusters)')
plt.show()
X_new=np.array([[13,2.5]])
X_new_scaled=scale.transform(X_new)
print(X_new_scaled)
print(kmeans.predict(X_new_scaled))

db=pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

plt.scatter(db['Fare'],db['Age'],c=db['Survived'])
plt.xlabel('Fare')
plt.ylabel('Age')
plt.plot([30,110],[0,80])
plt.show()

db['male']=db['Sex']=='male'
print(db[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values)
print(db['Survived'].values)

X=db[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y=db['Survived'].values

modelo=LogisticRegression()
modelo.fit(X,y)
print(modelo.coef_,modelo.intercept_)
print(modelo.predict([[3,True,22.0,1,0,7.25]]))
print(modelo.predict(X[:5]))
print(y[:5])
print(modelo.score(X,y))
y_predi=modelo.predict(X)
print(accuracy_score(y,y_predi))

FuckCancer=load_breast_cancer()
print(FuckCancer.keys())
print(FuckCancer['DESCR'])
print(FuckCancer['data'].shape)

dz=pd.DataFrame(FuckCancer['data'],columns=FuckCancer['feature_names'])
dz['target']=FuckCancer['target']
print(dz.head())

X=dz[FuckCancer.feature_names].values
y=dz['target'].values

modeli=LogisticRegression(solver='liblinear')
modeli.fit(X,y)
y_preda=modeli.predict(X)

print(modeli.predict([X[0]]))
print('accuracy score:',accuracy_score(y,y_preda))
print('precision score:',precision_score(y,y_preda))
print('recall score:',recall_score(y,y_preda))
print('f1 score:',f1_score(y,y_preda))
print(confusion_matrix(y,y_preda))

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)

print('whole sample size:',X.shape,y.shape)
print('Training set:',X_train.shape,y_train.shape)
print('Testing set:',X_test.shape,y_test.shape)

modeni=LogisticRegression(solver='liblinear')
modeni.fit(X_train,y_train)
print(modeni.score(X_test,y_test))
y_tema=modeni.predict(X_test)
def specificity_score(y_tema,y_test):
    p,r,t,y=precision_recall_fscore_support(y_tema,y_test)
    return r[0]
print('Accuracy score:',accuracy_score(y_tema,y_test))
print('Precision score:',precision_score(y_tema,y_test))
print('Recall/sensitivity score:',recall_score(y_tema,y_test))
print('F1 score:',f1_score(y_tema,y_test))
print('specificity score:',specificity_score(y_tema,y_test))
