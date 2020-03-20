import numpy as np
from . import solvers

class LogisticRegression:
    """
    Parameters
    ----------
    l1: float
        Used to specify the L1 regression penalty
    l2: float
        Used to specify the L2 regression penalty
    solver: str, {'gd', 'sgd', 'saga', 'svrg'}
        Optimization method
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG, 
        one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA, 
        one iteration uses only one data point.

    Attributes
    ----------
    coef_: array
        Coefficient of the features in the decision function.
    """

    SOLVER_SELECTOR = {'gd'         : solvers.GD,
                       'sgd'        : solvers.SGD,
                       'saga'       : solvers.SAGA,
                       'svrg'       : solvers.SVRG,
                       'mem_sgd'    : solvers.MEM_SGD}

    def __init__(self, solver='gd', l1=0.0, l2=0.05, max_iter=100, configs = None):
        
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter
        self.configs = configs

        self.solver = LogisticRegression.SOLVER_SELECTOR.get(solver, None)
        if self.solver == None:
            raise BaseException("Invalid Solver")

    def fit(self, A, b):
        """
        To fit the model to the data (A,b),
        we minimise the empirical risk over x:

            1/n \sum_{i=1}^n LogisticLoss(b_i, A[i]^T x) + l2/2 * ||x||^2,  
        
        where l2 is the L2 regularization coefficient.
        """
        
        # Dataset size
        n = len(b)

        # Smoothness constant
        L = 0.25 * max(np.linalg.norm(A,2,axis=1))**2 + self.l2

        # Strong convexity constant
        mu = self.l2
        
        # Initialisation of the solver
        x0=np.zeros(A.shape[1])
        
        def grad(x, i=None):
            '''
            Gradient of the objective function at x.
            If no i is given, grad should return the batch gradient.
            Otherwise, it should return the partial gradient associated the the datapoint (A[i], b[i]).
            '''
            
            if i is None: # return batch gradient
                output = -b/(1 + np.exp(b*np.dot(A, x)))
                output = sum(np.diag(output))/n @ A
                output += self.l2*x
            else: # return partial gradient associated the datapoint (A[i], b[i])
                output = -b[i] / (1 + np.exp(b[i] * np.dot(A[i], x)))
                output = output * A[i]
                output += self.l2 * x
                pass

            return output
        
        def prox(x, stepsize):
            return np.sign(x) * np.max(np.abs(x) - self.l1 * stepsize, 0)
        
        # Last iterate of the optimization, and table of all the iterates
        if self.configs != None:
            self.coef_, self._coef_tab = self.solver(x0, grad, self.max_iter, n, L, mu, self.configs)
        else:
            self.coef_, self._coef_tab = self.solver(x0, grad, self.max_iter, n, L, mu)
        self._empirical_risk = lambda x: sum(self._logistic_loss(A, b, x))/n \
                                 + self.l2/2.0 * np.linalg.norm(x,2)**2 \
                                 + self.l1 * np.linalg.norm(x,1)

    def decision_function(self, A):
        """
        Predict confidence scores for the samples in A.
        """
        if not hasattr(self, "coef_"):
            raise BaseException("The model hasn't been fitted yet.")

        return 1.0/(1 + np.exp(-np.dot(A, self.coef_)))

    def predict(self, A):
        """
        Predict class labels for samples in A.
        """
        if not hasattr(self, "coef_"):
            raise BaseException("The model hasn't been fitted yet.")

        return [[-1, 1][p >= 0.5] for p in self.decision_function(A)]

    def score(self, A, b):
        """
        Returns the mean accuracy on the given test data and labels.
        """
        return np.sum(self.predict(A) == b)/len(b)


    # _____________ private functions _____________

    def _logistic_loss(self, A, b, x):
        return np.log(1 + np.exp(-b*np.dot(A, x)))

