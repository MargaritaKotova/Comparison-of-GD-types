import numpy as np
import random

def GD(x0, grad, max_iter, n, L, mu):
    '''
    Parameters
    ----------
    x0: array, shape (nb_features,)
        Initialisation of the solver
    grad: function
        Gradient of the objective function
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG, 
        one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA, 
        one iteration uses only one data point.
    n: int
        Dataset size
    L: float
        Smoothness constant of the objective function
    mu: float
        Strong convexity constant of the objective function
        
    Returns
    -------
    x: array, shape (nb_features,)
        final iterate of the solver
    x_tab: array, shape (nb_features, max_iter)
        table of all the iterates
    '''
    stepsize  = 2/(mu+L)
    x         = x0
    x_tab     = np.copy(x)

    for k in range(max_iter):
        x     = x - stepsize*grad(x)
        x = prox(x, step_size)
        x_tab = np.vstack((x_tab, x))

    return x, x_tab

def SGD(x0, grad, max_iter, n, L, mu):
    step_size = 1/(L)
 
    x = x0
    x_tab = np.copy(x)

    for k in range(max_iter):
        i = np.random.randint(0, n)
        x = x - step_size * grad(x, i)
        x = prox(x, step_size)
        
        if k%n == 0: # each completed epoch
            x_tab = np.vstack((x_tab, x))

    return x, x_tab

def SAGA(x0, grad, max_iter, n, L, mu):
    grad_history = []
    step_size = 2/(mu+L)
    x = x0
    x_tab = np.copy(x)

    for k in range(max_iter):
        i = np.random.randint(0, n)
        grad_xi=grad(x, i)
        if len(grad_history)>0:
            x = x - step_size* (grad_xi - grad_history[-1] + np.mean(grad_history, axis=0))
            x = prox(x, step_size)
        grad_history.append(grad_xi)
        step_size = 1 / (2*(mu * len(grad_history) + L))
        
        if k%n == 0: # each completed epoch
            x_tab = np.vstack((x_tab, x))

    return x, x_tab


def SVRG(x0, grad, max_iter, n, L, mu):
    step_size = 2/(mu+L)
    x = x0
    x_tab = np.copy(x)
    M=10

    for k in range(max_iter):
        
        y=np.copy(x)

        for m in range(M):
            i = np.random.randint(0, n)
            y -= step_size * (grad(y, i) - grad(x, i) + grad(x))

        x=y
        x = prox(x, step_size)
        x_tab = np.vstack((x_tab, x))

    return x, x_tab

def MEM_SGD(x0, grad, max_iter, n, L, mu, configs = None):
    '''
    Parameters
    ----------
    x: array, shape (nb_features,)
        The point to apply the operation to
    k: number of coordinates to consider
    
    Returns
    -------
    x: array, shape (nb_features,)
        The final contracted view of the point
    '''
    def top_k(x, k):
        n = len(x)
        indices = x.argsort()[-k:][::-1]
        x_out = [x[i] if i in indices else 0 for i in range(0, n)]
        return x_out

    def rand_k(x, k):
        n = len(x)
        indices = random.sample(range(0, n), k)
        x_out = [x[i] if i in indices else 0 for i in range(0, n)]
        return x_out
    
    COMPRESSION_OPERATOR = {
        'top_k': top_k,
        'rand_k': rand_k
    }
    
    x0 = x0 + 1 # important step to avoid the bad approx. problem with the top_k comp
    if configs is None:
        raise BaseException("No configuration for the algorithm")

    comp = configs['comp']
    k = int(x0.shape[0] * configs['k'])
    print("Number of coefficients in percents: " + str(configs['k'])+", in samples: " +str(k) + ", total: " + str(x0.shape[0]))
    comp_operator = COMPRESSION_OPERATOR.get(comp, None)

    if comp_operator == None:
        raise BaseException("Invalid Compression Operator")
    x = x0
    x_tab = np.copy(x)
    d = len(x)
    m = np.zeros(x0.shape)
    x_avg = np.zeros(x0.shape)
    w_s = 0
    for t in range(0, max_iter):
        a = 10000*d/k
        step_size = 8/(mu*(t + a))

        i = np.random.randint(0, n)
        gradient = grad(x, i)
        
        g = comp_operator(m + step_size * gradient, k)

        x = x - g
        m = m + step_size * gradient - g

        w = (t + a)**2
        w_s = w_s + w
        x_avg = x_avg + w * np.copy(x)
      
        if t%(n) == 0: # each completed epoch
            x_avg = x_avg / w_s 
            x_tab = np.vstack((x_tab, np.copy(x_avg)))
            x_avg = np.zeros(x0.shape)
            w_s = 0

    return x, x_tab
    