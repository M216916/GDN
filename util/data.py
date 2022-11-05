# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile


def get_attack_interval(attack): 
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i-1] == 0:
                heads.append(i)
            
            if i < len(attack)-1 and attack[i+1] == 0:
                tails.append(i)
            elif i == len(attack)-1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res
