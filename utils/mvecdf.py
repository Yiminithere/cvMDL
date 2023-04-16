import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


class MvECDF:
    def __init__(self, data):
        """
        initialize the MvECDF class
        
        parameters:
        -----------
        data: either a list or an array to represent the set of points for which ECDF is calculated

        """
        self._data = np.array(data)
        self._dimension = self._data.shape
        if len(self._dimension) > 2:
            raise TypeError("Incorrect input data dimension.") 
        elif len(self._dimension) == 1 or self._dimension[1] == 1:
            self._ecdf = ECDF(self._data.flatten())
        else:
            self._argsort = {i: np.argsort(self._data[:,i]) for i in range(self._dimension[1])}
            self._sorted = {i: self._data[:,i][self._argsort[i]] for i in range(self._dimension[1])}

    def random_match(self, x, y):
        """
        Finds the number of elements in an increasing sequence y that 
        are less than or equal to x

        """
        n = len(y)
        if x < y[0]:
            return 0
        if x >= y[-1]:
            return n
        m = np.random.randint(0,n-1)
        if x <= y[m]:
            return self.random_match(x, y[:(m+1)])
        elif x > y[m]:
            return m + 1 + self.random_match(x, y[(m+1):])
          
    def ecdf_eval(self, x):
        """
        Evaluates the empirical CDF of the input data at x

        """
        if type(x) != list or type(x) != np.ndarray:
            x = [x]
        x = np.array(x).flatten()
        if len(self._dimension) == 1 or self._dimension[1] == 1:
            if len(x) == 1:
                return self._ecdf(x.flatten())[0]
            else:
                raise TypeError("Inconsistent input data dimension.")
        else: 
            if len(x) != self._dimension[1]:
                raise TypeError("Inconsistent input data dimension.")
            else:
                A = np.zeros((len(x),self._dimension[0]))
                for i in range(len(x)):
                    ind = self.random_match(x[i], self._sorted[i])
                    if ind == 0:
                        return 0
                    else:
                        A[i,:][self._argsort[i][:ind]] = 1
                return np.sum(np.prod(A, axis = 0))/self._dimension[0]