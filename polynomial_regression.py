#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values  

from sklearn.linear_model import LinearRegression
r1=LinearRegression()
r1.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
r2=PolynomialFeatures(degree =2)
X_poly = r2.fit_transform(X)
lin_reg2= LinearRegression()
lin_reg2.fit(X_poly,Y)


plt.scatter(X,Y,color='red')
plt.plot(X,r1.predict(X),color='blue')
plt.title("Linear Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y,color='red')
plt.plot(X_grid,lin_reg2.predict(r2.fit_transform(X_grid)),color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
