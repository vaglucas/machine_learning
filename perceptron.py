import numpy as np
class Perceptron(object):
    """Perceptron classifier

    Parameters
    -----
    eta : float
        learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset

    Attributes
    ----------
    w_ : id-array
        weights after fitting
    errors_ : list
        number of miclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X,y):
        """fit training data.
        #Parameters
        X : {array-like}, shape = {n_samples, n_features}
            #trining vectors, where n_sample is the number
            #of samples an n_features is the number of features
        y : array-like, shape[n_samples]
            #target values
        #returns
        self : object
        """
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors= 0
            for xi, target in zip(X,y):
                update = self.eta = (target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update!= 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """calculate net input"""
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0,1,-1)
