
import Hybrid_model as Hm
import warnings
from datetime import datetime
from sampen import sampen2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
import torch
from Get_dataset import get_dataset
from PyEMD import EMD, EEMD, CEEMDAN  # For module 'PyEMD', please use 'pip install EMD-signal' instead.
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as st
import numpy as np



"""
================================================================================================================================================================================================================================================================================================================================================================================================================
                                                            
================================================================================================================================================================================================================================================================================================================================================================================================================
"""


def standardize_multivariate_ts(A: pd.DataFrame, B: pd.DataFrame):
    
    mean_vals = B.mean()
    std_vals = B.std()

    A_standardized = (A - mean_vals) / std_vals

    return A_standardized, mean_vals, std_vals


def inverse_standardize_multivariate_ts(A_standardized: pd.DataFrame, mean_vals: pd.Series, std_vals: pd.Series):
    
    A_original = A_standardized * std_vals + mean_vals
    return A_original


def plot_corr(ts_data,target_num, max_lag=None):
    
    if max_lag is None:
        max_lag = len(ts_data) // 2

    
    target_ts = ts_data.iloc[:,target_num]

    #
    target_acf = st.acf(target_ts, nlags=max_lag, fft=False)
    
    
    conf_int = 1.96/np.sqrt(len(target_ts))
    
    plt.figure()
    plt.bar(range(len(target_acf)), target_acf)
    plt.axhline(y=conf_int, color='r', linestyle='--')
    plt.axhline(y=-conf_int, color='r', linestyle='--')
    plt.title("Autocorrelation of target time series")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.show()
    
    
    above_conf_int = np.abs(target_acf) > conf_int
    if np.sum(above_conf_int) / len(target_acf) > 0.05:
        print("The autocorrelation function is tailing off slowly, indicating a non-stationary time series.")
    else:
        print("The autocorrelation function cuts off after some lags, indicating a stationary time series.")
    
    cross_corr_values = []

    
    for i in range(1, ts_data.shape[1]):
        other_ts = ts_data.iloc[:,i]
        cross_corr = st.ccf(target_ts, other_ts)[:max_lag+1]
        cross_corr_values.append(cross_corr)
        
        plt.figure()
        plt.bar(range(len(cross_corr)), cross_corr)
        plt.axhline(y=conf_int, color='r', linestyle='--')
        plt.axhline(y=-conf_int, color='r', linestyle='--')
        plt.title(f"Cross correlation between target and variable {i}")
        plt.xlabel("Lag")
        plt.ylabel("Cross-correlation")
        plt.show()

    
    return target_acf, cross_corr_values

"""
sliding window
"""
import pandas as pd

def sliding_window(data, N, W):

    if N <= 0 or W <= 0:
        raise ValueError("N and W must >0 and be an integer.")

    O1 = []  
    O2 = []  

    for i in range(len(data) - W - N + 1):
        L1_window = data.iloc[i:i + W + N]
        L2_window = data.iloc[i:i + W + N + 1]
        if len(L2_window) == N + W + 1:
            O1.append(L1_window)
            O2.append(L2_window)
        else:
            break

# # example
# data = {'A': [1, 2, 3, 4, 5, 6, 7], 'B': [6, 7, 8, 9, 10, 11, 12], 'C': [11, 12, 13, 14, 15, 16, 17]}
# signal_df = pd.DataFrame(data)
# N = 2
# W = 2
# O1, O2 = sliding_window(signal_df, N, W)



    return O1, O2



