import numpy as np


def insample_ssq(y):
    return np.var(y)*len(y)



def mse(X, dt_Y, d, split):
    
    before_mse = insample_ssq( dt_Y )
    after_mse = insample_ssq( dt_Y[ X[:,d] < split ] ) + insample_ssq( dt_Y[ X[:,d] >= split ] )
    
    return (before_mse - after_mse) / len(dt_Y)


def gini(X, dt_Y, d, split):
    return None