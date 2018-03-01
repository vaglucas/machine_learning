import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from adaline import ADalineGD
from adalineSGD import AdalineSGD

class functionUtility(object):

    def plot_dexision_regions(X,y, classifier, resolution=0.02):
        #setup maker generator and color map
        markers = ('s','x','o','^','v')
        colors = ('red','blue','lightgreen','grey','cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        #plot the decision surface
        x1_min, x1_max = X[:,0].min()-1,X[:,0].max()+1
        x2_min, x2_max = X[:,1].min()-1,X[:,1].max()+1

        xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
        Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())

        #plot class n_samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
