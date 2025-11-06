
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
import os
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

warnings.filterwarnings("ignore")  # Ignore some annoying warnings

# The default dataset saving path: D:\\CEEMDAN_LSTM\\

PATH = '/data/ajiong/MEMD_predict/'  # (Server load)
# The default figures saving path: D:\\CEEMDAN_LSTM\\figures\\
FIGURE_PATH = PATH + 'figures/'
# The default logs and output saving path: D:\\CEEMDAN_LSTM\\subset\\
LOG_PATH = PATH + 'subset/'
# The default dataset name of a csv file: cl_sample_dataset.csv (must be csv file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for a multivariate time series DataFrame.
    
    Parameters:
    - df: DataFrame of the multivariate time series
    
    Returns:
    - corr_matrix: Correlation matrix of the multivariate time series
    """
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Multivariate Time Series')
    plt.show()
    
    return corr_matrix

"""
================================================================================================================================================================================================================================================================================================================================================================================================================
                                                        Data processing
================================================================================================================================================================================================================================================================================================================================================================================================================
"""



#decompose the data
DATASET_NAME = 'ETTm2_all'
Hm.declare_vars(mode = 'emd')

#change the target var to first column, and set the order = 0
all_data =  get_dataset(type='ETTm2_all')     #  get_dataset(type='wind_speed')


all_data.loc[:, ['HUFL', 'OT']] = all_data.loc[:, ['OT','HUFL']].to_numpy()
all_data.columns = ['OT','HULL', 'MUFL', 'MULL', 'LUFL', 'LULL','HUFL' ]
SERIES = all_data.iloc[:,0]
SERIES.name = 'ETTm2_all'
Hm.declare_path(path=PATH, figure_path=FIGURE_PATH, log_path=LOG_PATH, dataset_name=DATASET_NAME, series=SERIES)


all_data = all_data.iloc[:5000,:]  





all_data = all_data.iloc[:5000,:] 



"setted Parameters"
# train_len = 
# valid_len = 
# test_len  = 
# windows = 

# predict_len =

look_back = windows
train_data, valid_data, test_data = all_data.iloc[:train_len,:],all_data.iloc[train_len:train_len+valid_len,:]  , all_data.iloc[train_len+valid_len:,:]

order= 0   #columns of var


"""
================================================================================================================================================================================================================================================================================================================================================================================================================
                                                            Proposed Methods
================================================================================================================================================================================================================================================================================================================================================================================================================
"""

from MULTI_RDEM_all_model import standardize_multivariate_ts,plot_corr,sliding_window,inverse_standardize_multivariate_ts

#Normalize
normalize_all_data, mean_vals, std_vals = standardize_multivariate_ts(all_data, train_data)

normalize_train_data, normalize_valid_data, normalize_test_data= normalize_all_data.iloc[:train_len,:],normalize_all_data.iloc[train_len:train_len+valid_len,:]  , normalize_all_data.iloc[train_len+valid_len:,:]





"""
=======================Method:MEMD================
"""


normalize_all_data_array = np.array(normalize_all_data)
#all_data_array = np.array(all_data)

from MEMD_all import memd

all_imf = memd(normalize_all_data_array)

np.save(SERIES.name+'_all_imfs.npy', all_imf)



loaded_imfs = np.load(SERIES.name+'_all_imfs.npy')  #SERIES.name+'No_normallize_all_imfs.npy'



all_imf = loaded_imfs

normalize_train_all_imf  = all_imf[:,:,:train_len]
normalize_valid_all_imf  = all_imf[:,:,train_len:train_len + valid_len]


normalize_test_all_imf   = all_imf[:,:,train_len+valid_len-look_back:]  


import seaborn as sns











"""
=======================Method:N-MA-EMD================
"""


def Roll_create_dateback_multivar(data, DATE_BACK, ahead=1, order=0): 
    
    # Normalize without unifying
    trainY = np.array(data)
    trainX = trainY

    # Create dateback
    dataX, dataY = [], []
    ahead = ahead - 1
    for i in range(len(trainY) - DATE_BACK - ahead):
        dataX.append(copy.deepcopy(np.array(trainX[i:(i + DATE_BACK),:])))
        dataY.append(copy.deepcopy(np.array(trainY[i + DATE_BACK + ahead,order])))

    return np.array(dataX), np.array(dataY), np.array(trainX[-DATE_BACK:])





all_varibales_imfs = []
Hm.declare_vars(mode = 'emd')
for i in range(normalize_all_data_array.shape[1]):
    waited_decom_TS = normalize_all_data_array[:,i]
    
    waited_decom_TS_imfs = Hm.emd_decom(series=pd.Series(waited_decom_TS), draw=False)
    all_varibales_imfs.append(waited_decom_TS_imfs)





def count_extrema(df):
    
    extrema_counts = {}
    
    
    for column in df.columns:
        data = df[column].to_numpy()  
        
        
        maxima = (data[:-2] < data[1:-1]) & (data[1:-1] > data[2:])
        minima = (data[:-2] > data[1:-1]) & (data[1:-1] < data[2:])
        
        
        total_extrema = np.sum(maxima) + np.sum(minima)
        
      
        extrema_counts[column] = total_extrema
    
    return extrema_counts


# Provide the functions without running them


averages_extrema_num_all = []

for imfs in all_varibales_imfs:


    averages_extrema_num = count_extrema(imfs)
    averages_extrema_num_all.append(averages_extrema_num)

for extrema_num in averages_extrema_num_all:
    print(extrema_num)


def group_imfs_by_distance(imfs_extrema_counts, order):
    """
    Group each IMF of non-target variables by their distance to target variable IMFs based on extrema counts.
    """
    target_imfs_counts = imfs_extrema_counts[order]
    
    groups = {i: [] for i in range(len(target_imfs_counts))}
    
    # Iterate over each variable's IMFs
    for var_idx, var_imfs_counts in enumerate(imfs_extrema_counts):
        if var_idx == order:  # Skip the target variable
            continue
        # Iterate over each IMF of the current variable
        for imf_idx, imf_idx_name in enumerate(var_imfs_counts): 
            imf_count = var_imfs_counts[imf_idx_name]
            min_distance = float('inf')
            min_distance_idx = -1
            # Compare with each IMF of the target variable
            for target_idx, target_idx_name in enumerate(target_imfs_counts):
                target_count = target_imfs_counts[target_idx_name]
                distance = abs(imf_count - target_count)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_idx = target_idx
            # Assign the current IMF to the closest target IMF group

            groups[min_distance_idx].append((var_idx, imf_idx))
    
    return groups

groups = group_imfs_by_distance(averages_extrema_num_all, order=order)






def combine_imfs_by_groups_v2(groups, all_varibales_imfs, order):
    grouped_data = []
    
    
    for key, group in groups.items():
        # Create a list to store IMFs for the current group
        
        #group like [(0, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
        current_group_data = []
        
        # Append the target variable IMFs first
        target_var_imfs = all_varibales_imfs[order]



        current_group_data.append(target_var_imfs.iloc[:, key].values)
        
        # Now append the related variable IMFs based on the group
        for var_idx, imf_idx in group:

            if var_idx != order:  # We've already added the target variable IMFs
                current_group_data.append(all_varibales_imfs[var_idx].iloc[:, imf_idx].values)
            
        print(len(current_group_data))

        temp_current_group_data = np.column_stack(current_group_data)
        

        temp_current_group_data[:, [0, order]] = temp_current_group_data[:, [order,0]]



        # Convert the list of IMFs to a 2D numpy array and add it to the grouped data
        grouped_data.append(temp_current_group_data)
        
    return grouped_data

grouped_IMFs_data_v2 = combine_imfs_by_groups_v2(groups, all_varibales_imfs, order)
grouped_IMFs_data_extreme = grouped_IMFs_data_v2





















"""

=======================Method:MA-EMD================

"""







partial_all_varibales_imfs = copy.deepcopy(all_varibales_imfs)
for i in range(len(averages_extrema_num_all)):
    print("Check extrema number of NO.%s个var decomposition result"%i)
    for site,nums in zip(range(len(averages_extrema_num_all[i].values())),averages_extrema_num_all[i].values()):
        if nums<=50:
            new_col = partial_all_varibales_imfs[i].iloc[:, site:].sum(axis=1)
            new_df = pd.concat([partial_all_varibales_imfs[i].iloc[:, :site], new_col], axis=1)
            new_df.columns = partial_all_varibales_imfs[i].columns[:site+1]
            partial_all_varibales_imfs[i] = copy.deepcopy(new_df)

            break


partial_averages_extrema_num_all = []

for imfs in partial_all_varibales_imfs:

    averages_extrema_num = count_extrema(imfs)
    partial_averages_extrema_num_all.append(copy.copy(averages_extrema_num))

for extrema_num in partial_averages_extrema_num_all:
    print(extrema_num)





T_IMF_all = copy.deepcopy(partial_all_varibales_imfs[order])



import numpy as np
import pandas as pd
from scipy.stats import entropy

def count_extrema_intervals_exact(data):

    extrema_info = {}
    for i in range(data.shape[1]):
        series = data[:, i]
        
        maxima = np.where((series[:-2] < series[1:-1]) & (series[1:-1] > series[2:]))[0] + 1
        minima = np.where((series[:-2] > series[1:-1]) & (series[1:-1] < series[2:]))[0] + 1
        
        extrema = np.sort(np.concatenate((maxima, minima)))
        
        intervals = np.diff(extrema)
        
        interval_counts = pd.Series(intervals).value_counts().sort_index()
        extrema_info[i] = interval_counts
    return extrema_info

def calculate_kl_divergence_and_group_imfs(all_variables_imfs, order, epsilon=0.000000001):

    
    t_imfs = all_variables_imfs[order]

    
    if not isinstance(t_imfs, pd.DataFrame):
        raise ValueError("t_imfs should be a pandas DataFrame.")

    
    t_imfs_distributions = [count_extrema_intervals_exact(t_imf.values.reshape(-1, 1)) for _, t_imf in t_imfs.iteritems()]

    
    all_intervals = sorted(set.union(*[set(dist[0].index) for dist in t_imfs_distributions]))

    
    extended_t_imfs_distributions = []
    for dist in t_imfs_distributions:
        extended_distribution = dist[0].reindex(all_intervals, fill_value=0)
        extended_distribution += epsilon
        extended_distribution /= extended_distribution.sum()
        extended_t_imfs_distributions.append(extended_distribution)

    groups = {i: [] for i in range(t_imfs.shape[1])}
    min_kl_divergences = {}  

    for var_idx, imfs in enumerate(all_variables_imfs):
        if var_idx == order:  
            continue

        for imf_idx, (column_name, r_imf_column) in enumerate(imfs.iteritems()):
            r_imf_extrema_info = count_extrema_intervals_exact(r_imf_column.values.reshape(-1, 1))[0]
            r_imf_distribution = r_imf_extrema_info.reindex(all_intervals, fill_value=0)
            r_imf_distribution += epsilon
            r_imf_distribution /= r_imf_distribution.sum()

            kl_divergences = [entropy(r_imf_distribution, t_imf_distribution) for t_imf_distribution in extended_t_imfs_distributions]
            min_kl_div_idx = np.argmin(kl_divergences)
            min_kl_div = kl_divergences[min_kl_div_idx]

            groups[min_kl_div_idx].append((var_idx, imf_idx))
            min_kl_divergences[(var_idx, imf_idx)] = min_kl_div  

    return groups, min_kl_divergences



partial_groups,min_kl_divergences = calculate_kl_divergence_and_group_imfs(partial_all_varibales_imfs, order, epsilon=0.000000001)



partial_grouped_IMFs_data_v2 = combine_imfs_by_groups_v2(partial_groups, partial_all_varibales_imfs, order)



"result is the grouped IMFs"

new_grouped_IMFs_data_extreme = partial_grouped_IMFs_data_v2


