import numpy as np

class ADelineGD(object):
    """ADAptive LInear NEuron classifier
    Parameters
    ----------
    eta: float
        learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset

    Attributes
    -----------
    w_ : 1d-array
        weights after fitting
    errors_ : List
        number of misclassfications in every epoch

    """


    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter


    def fit(self, X, y):
        """fit training data.
        #Parameters
        X : {array-like}, shape = {n_samples, n_features}
            trining vectors, where n_sample is the number
            of samples an n_features is the number of features
        y : array-like, shape[n_samples]
            target values
        #returns
        self : object
        """

        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ calculate net input"""
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(sefl, X):
        """compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """return class label after unit setp"""
        return np.where(self.activation(X)>=0.01,1,-1)