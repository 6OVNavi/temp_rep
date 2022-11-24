import numpy as np
import matplotlib.pyplot as plt

X=np.array(range(100))#+np.random.uniform(0,1)
y=np.array(range(100))

X=X.reshape(-1,1)

from sklearn.linear_model import SGDRegressor,LinearRegression

SGD=SGDRegressor(penalty='l2',learning_rate='constant',eta0=0.01,alpha=0.01)
lr=LinearRegression()

SGD.fit(X,y)
lr.fit(X,y)

Xval=np.array(range(115,200))#+np.random.uniform(0.001,0.01)
yval=np.array(range(115,200))

Xval=Xval.reshape(-1,1)

from sklearn.metrics import r2_score

sgd_pred=SGD.predict(Xval)
lr_pred=lr.predict(Xval)
print(sgd_pred)
print(r2_score(yval,sgd_pred))
print(r2_score(yval,lr_pred))
