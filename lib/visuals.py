import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style='darkgrid',
        palette='bright', font='sans-serif',
        font_scale=2, color_codes=True)

def learning_curve(*models, **named_models):
    """
    Plot the empirical risk curves wrt the number of epochs for all the models inputed.
    The models can be passed as follows:
        learning_curve(model0, model1, name2=model2, name3=model3)
    where the `names` are display names.
    """
    plt.figure(figsize=(20,10))
    if models:
        for model in models:
            plt.plot(get_curve(model), linewidth=5.0, linestyle="-")

    if named_models:
        for name, model in named_models.items():
            plt.plot(get_curve(model), linewidth=5.0, linestyle="-", label=name)

        plt.grid(True)
        plt.legend(loc='best')
        plt.ylabel('Empirical Risk')
        plt.xlabel('Epochs')
        plt.title('Learning curves')

    plt.show()


def get_curve(model):
    """
    Get the empirical risk values for each epoch
    """
    if not hasattr(model, "coef_"):
        raise Exception("The model hasn't been fitted yet.")

    x_tab = model._coef_tab

    curve = [model._empirical_risk(x) for x in x_tab]
    return curve

    
def l1_regularization_plot(clf, A, b):
    """
    Plot showing the effect of L1 regularization
    """
    plt.figure(figsize=(20,10))
    l1_tab = np.power(2.0, np.array(range(-12,1)))

    for l1 in l1_tab:
        clf_ = clf(l1)
        clf_.fit(A, b)
        coef = clf_.coef_
        for j in range(len(coef)):
            if np.abs(coef[j])>1e-14:
                plt.plot(l1, j  , 'o', color='blue', markersize=12)
                
    plt.grid(False)
    plt.ylabel('Non-null coordinates of x')
    plt.xlabel('log(l1)')
    plt.xscale('log')
    plt.ylim(-1,len(coef))
    plt.yticks(np.arange(0,len(coef)))
    plt.title('Sparsity of the solution for various values of l1')
    plt.show()
    
def l2_regularization_plot(clf, A, b):
    """
    Plot showing the effect of L1 regularization
    """
    plt.figure(figsize=(20,10))
    l2_tab = np.power(2.0, np.array(range(-8,6)))
    
    coefs = []

    for i, l2 in enumerate(l2_tab):
        clf_ = clf(l2)
        clf_.fit(A, b)
        
        coefs += [clf_.coef_]
        
    coefs = np.array(coefs)
    plt.plot(coefs, linewidth=5)
              
    plt.grid(False)
    plt.ylabel('Value of the different coordinates of x')
    plt.xlabel('log(l2)')
    plt.xscale('log')
    plt.title('Amplitude of the coefficients of the solution for various values of l2')
    plt.show()