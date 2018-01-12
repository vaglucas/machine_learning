from numpy.random import seed


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier

    Parameters
    ------------
    eta : float
        Learnin rate between  0 and 1
    n_iter : int
        passes over the training dataset


    Attributes
    ------------------
    w_ : id_array
        weights after fitting
    errors_ : List
        Number of misclassfications in every epoch
    shuffle : bool
        Shuffles training data every epoch
        if true to prevent cycles
    random_state : int (default none)
        set random state for shuffing
        and initializing th weights

    """

    def __init__(self, eta=0.01,n_iter=10,shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self,X,y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initializa_weights(X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            if self.shuffle:
                X, y= self._shuffle(X,y)
            cost=[]
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            selg.cost_.append(avg_cost)
        return self
    
