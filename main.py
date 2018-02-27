import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from adaline import ADalineGD
from adalineSGD import AdalineSGD
from functionUtility import functionUtility as fc
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

print("=======================DATI=========================")

y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values


plt.scatter(X[:50,0],X[:50,-1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('sepal length d')
plt.ylabel('petal length d')
plt.legend(loc='upper left')
plt.show()
print("=======================DATI=========================")



print("=======================Perceptron=========================")

ppn = Perceptron(eta=0.01,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs perceptron')
plt.ylabel('Number of misclassfications perceptron')
plt.tight_layout()

plt.show()

fc.plot_dexision_regions(X,y,classifier=ppn)
plt.xlabel('sepal length [cm]ppc')
plt.ylabel('petal length [cm]ppc')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()

print("=======================Perceptron=========================")







"""ADaline"""
print("=======================ADALINE=========================")

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

ada1 = ADalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(1,len(ada1.cost_)+1,np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-square-error)')
ax[0].set_title('Adaline Learning rate 0.01')

ada2 = ADalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(1,len(ada2.cost_)+1,np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(sum-square-error)')
ax[1].set_title('Adaline Learning rate 0.0001')


X_std = np.copy(X)
X_std[:,0] =(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] =(X[:,1]-X[:,1].mean())/X[:,1].std()

ada = ADalineGD(n_iter=15,eta=0.01)
ada.fit(X_std,y)
fc.plot_dexision_regions(X_std,y, classifier=ada)
plt.title('Adaline Gradient Descent')
plt.xlabel('sepal length[standardized]DG')
plt.xlabel('petal length[standardized]DG')
plt.legend(loc='upper left DG')
plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_, marker='o')
plt.xlabel('Epochs DG')
plt.ylabel('sum-squared-error DG')
plt.tight_layout()
plt.show()
print("=======================ADALINE=========================")

print("=======================SDG=========================")

adaSDG = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
adaSDG.fit(X_std, y)

fc.plot_dexision_regions(X_std, y, classifier=adaSDG)
plt.title('AdalineSDG - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]SDG')
plt.ylabel('petal length [standardized]SDG')
plt.legend(loc='upper left SDG')
plt.tight_layout()
#plt.savefig('./adaline_4.png', dpi=300)
plt.show()

plt.plot(range(1, len(adaSDG.cost_) + 1), adaSDG.cost_, marker='o')
plt.xlabel('Epochs SDG')
plt.ylabel('Average Cost SDG')

plt.tight_layout()
# plt.savefig('./adaline_5.png', dpi=300)
plt.show()
print("=======================SDG=========================")
