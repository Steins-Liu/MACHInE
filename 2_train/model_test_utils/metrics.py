import numpy as np

def mean_absolute_relative_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.mean(np.abs((y_true - y_pred) / y_true))

def coefficient_of_determination(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - np.mean(y_true)) ** 2).sum()
    R2_val = 1-u/v
    return R2_val

def discriminator_score(y_true, y_pred):
    i = 0
    total = 0
    while i < len(y_true):
        total = total + np.abs(y_true[i] - y_pred[i])	
        i = i+1	
    return (total / len(y_true))
