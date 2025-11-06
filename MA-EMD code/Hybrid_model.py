#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division, print_function
from calendar import c
from curses import def_shell_mode
from pickle import NONE
import re
from tkinter.messagebox import NO
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
#warnings.filterwarnings("ignore")  # Ignore some annoying warnings

__version__ = '1.0.0a'
__module_name__ = 'Hybrid_model'
print('Importing...', end='')
# 0.Introduction
# ==============================================================================================
# This is a initialization module for a complete CEEMDAN-LSTM forecasting process.
# This is a module still testing. Some errors may occur during runtime.
# The following modules need to be installed before importing at Anaconda3.
# Otherwise, please install the corresponding modules by yourself according to the warnings.
# pip install EMD-signal
# pip install sampen
# pip install vmdpy
# pip install datetime
# pip install tensorflow-gpu==2.5.0
# pip install scikit-learn


# import CEEMDAN_LSTM as cl


# Contents
# ==============================================================================================
# 0.Introduction
# 1.Guideline functions
# 2.Declare default variables
# 3.Decomposition, Sample entropy, Re-decomposition, and Integration
# 4.LSTM Model Functions
# 5.CEEMDAN-LSTM Forecasting Functions
# 6.Hybrid Forecasting Functions
# 7.Statistical Tests
# Appendix if main run example
# ==============================================================================================

# Import basic modules
# More modules will be imported before the corresponding function
# import logging # logger = logging.getLogger(__name__)
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
warnings.filterwarnings("ignore")  # Ignore some annoying warnings
from datetime import datetime

# Import module for EMD decomposition
# It is the EMD-signal module with different name to import
from PyEMD import EMD, EEMD, CEEMDAN  # For module 'PyEMD', please use 'pip install EMD-signal' instead.

# Import module for sample entropy
#from sampen import sampen2

# Import modules for LSTM prediciton
# Sklearn
from sklearn.preprocessing import MinMaxScaler  # Normalization
from sklearn.metrics import r2_score  # R2
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import mean_absolute_percentage_error  # MAPE
# Keras

# Statistical tests
from statsmodels.tsa.stattools import adfuller  # adf_test
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test  # LB_test
from statsmodels.stats.stattools import jarque_bera as jb_test  # JB_test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # plot_acf_pacf
import tensorflow as tf
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

# -------------------------------
# The default dataset saving path: D:\\CEEMDAN_LSTM\\
PATH = 'D:\\CEEMDAN_LSTM\\'
# The default figures saving path: D:\\CEEMDAN_LSTM\\figures\\
FIGURE_PATH = PATH + 'figures\\'
# The default logs and output saving path: D:\\CEEMDAN_LSTM\\subset\\
LOG_PATH = PATH + 'subset\\'
# The default dataset name of a csv file: cl_sample_dataset.csv (must be csv file)
DATASET_NAME = 'cl_sample_dataset'
# The default time series dataset. Load from DATASET_NAME or input a pd.Series.
SERIES = None


# Files variables declare functions
# -------------------------------
# Declare the path
# You can also enter the time series data directly by declare_path(series)
def declare_path(path=PATH, figure_path=FIGURE_PATH, log_path=LOG_PATH, dataset_name=DATASET_NAME, series=SERIES):
    # Check input
    global PATH, FIGURE_PATH, LOG_PATH, DATASET_NAME, SERIES
    for x in ['path', 'figure_path', 'log_path', 'dataset_name']:
        if type(vars()[x]) != str: raise TypeError(x + ' should be strings such as D:\\\\CEEMDAN_LSTM\\\\...\\\\.')
    if path == '' or figure_path == '' or log_path == '':
        raise TypeError('PATH should be strings such as D:\\\\CEEMDAN_LSTM\\\\...\\\\.')
    # declare FIGURE_PATH,LOG_PATH if user only inputs PATH or inputs them at different folders

    # Change path
    ori_figure_path, ori_log_path = FIGURE_PATH, LOG_PATH
    if path != PATH:
        # Fill path if lack like 'PATH=D:\\CEEMDAN_LSTM' to 'PATH=D:\\CEEMDAN_LSTM\\'
        if path[-1] != '\\': path = path + '\\'
        PATH = path
        FIGURE_PATH, LOG_PATH = PATH + 'figures\\', PATH + 'subset\\'
    if figure_path != ori_figure_path:
        if figure_path[-1] != '\\': figure_path = figure_path + '\\'
        FIGURE_PATH = figure_path  # Separate figure saving path
    if log_path != ori_log_path:
        if log_path[-1] != '\\': log_path = log_path + '\\'
        LOG_PATH = log_path  # Separate log saving path
    DATASET_NAME, SERIES = dataset_name, series  # update variables

    # Check or create a folder for saving
    print('Saving path: %s' % PATH)
    for p in [PATH, FIGURE_PATH, LOG_PATH]:
        if not os.path.exists(p): os.makedirs(p)

    # Check whether inputting a series
    if SERIES is not None:
        if not isinstance(series, pd.Series):
            raise ValueError('The inputting series must be pd.Series.')
        else:
            print('Get input series named:', str(series.name))
            SERIES = series.sort_index()  # sorting

    # Load Data for csv file
    else:
        # Check csv file
        if not (os.path.exists(PATH + DATASET_NAME + '.csv')):
            raise ImportError(
                'Dataset is not exists. Please input dataset_name=' + DATASET_NAME + ' and check it in: ' + PATH
                + '. You can also input a pd.Series directly.')
        else:
            print('Load sample dataset: ' + DATASET_NAME + '.csv')
            # Load sample dataset
            df_ETS = pd.read_csv(PATH + DATASET_NAME + '.csv', header=0, parse_dates=['date'],
                                 date_parser=lambda x: datetime.strptime(x, '%Y%m%d'))

            # Select close data and convert it to time series data
            if 'date' not in df_ETS.columns or 'close' not in df_ETS.columns:
                raise ValueError(
                    "Please name the date column and the required price column as 'date' and 'close' respectively.")
            SERIES = pd.Series(df_ETS['close'].values, index=df_ETS['date'])  # 选择收盘价
            SERIES = SERIES.sort_index()  # sorting

    # Save the required data to avoid chaanging the original data
    pd.DataFrame.to_csv(SERIES, PATH + 'demo_data.csv')

    # Show data plotting
    fig = plt.figure(figsize=(10, 4))
    SERIES.plot(label='Original data', color='#0070C0')  # F27F19 orange #0070C0 blue
    plt.title('Original Dataset Figure')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(FIGURE_PATH + 'Original Dataset Figure.svg', bbox_inches='tight')
    plt.show()

    return SERIES  # pd.Series


# Model variables
# -------------------------------
# Mainly determine the decomposition method
MODE = 'ceemdan'
# Integration form only effective after integration
FORM = ''  # such as '233' or 233
# The number of previous days related to today
DATE_BACK = 30
# The length of the days to forecast
PERIODS =    4110         #   4111         #320
# LSTM epochs
EPOCHS = 200
# Patience of adaptive learning rate and early stop, suggest 1-20
PATIENCE = 30


# Declare model variables
def declare_vars(mode=MODE, form=FORM, data_back=DATE_BACK, periods=PERIODS, epochs=EPOCHS, patience=None):
    print('##################################')
    print('Global Variables')
    print('##################################')

    # Change and Check
    global MODE, FORM, DATE_BACK, PERIODS, EPOCHS, PATIENCE
    FORM = str(form)
    MODE, DATE_BACK, PERIODS, EPOCHS = mode.lower(), data_back, periods, epochs
    if patience is None:
        PATIENCE = int(EPOCHS / 10)
    else:
        PATIENCE = patience
    check_vars()

    # Show
    print('MODE:' + str.upper(MODE))
    print('FORM:' + str(FORM))
    print('DATE_BACK:' + str(DATE_BACK))
    print('PERIODS:' + str(PERIODS))
    print('EPOCHS:' + str(EPOCHS))
    print('PATIENCE:' + str(PATIENCE))


# Check the type of model variables
def check_vars():
    global FORM
    if MODE not in ['emd', 'eemd', 'ceemdan', 'emd_se', 'eemd_se', 'ceemdan_se']:
        raise TypeError('MODE should be emd,eemd,ceemdan,emd_se,eemd_se,or ceemdan_se rather than %s.' % str(MODE))
    if not type(FORM) == str:
        raise TypeError('FORM should be strings in digit such as 233 or "233" rather than %s.' % str(FORM))
    if not (type(DATE_BACK) == int and DATE_BACK > 0):
        raise TypeError('DATE_BACK should be a positive integer rather than %s.' % str(DATE_BACK))
    if not (type(PERIODS) == int and PERIODS >= 0):
        raise TypeError('PERIODS should be a positive integer rather than %s.' % str(PERIODS))
    if not (type(EPOCHS) == int and EPOCHS > 0):
        raise TypeError('EPOCHS should be a positive integer rather than %s.' % str(EPOCHS))
    if not (type(PATIENCE) == int and PATIENCE > 0):
        raise TypeError('PATIENCE should be a positive integer rather than %s.' % str(PATIENCE))
    if FORM == '' and (MODE in ['emd_se', 'eemd_se', 'ceemdan_se']):
        raise ValueError('FORM is not delcared. Please delcare is as form = 233 or "233".')


# Check dataset input a test one or use the default one
# -------------------------------
def check_dataset(dataset, input_form, no_se=False, use_series=False,
                  uni_nor=False):  # uni_nor is using unified normalization method or not
    file_name = ''
    # Change MODE
    global MODE
    if no_se:  # change MODE to the MODE without se
        check_vars()
        if MODE[-3:] == '_se':
            print('MODE is', str.upper(MODE), 'now, using %s instead.' % (str.upper(MODE[:-3])))
            MODE = MODE[:-3]
    # Use SERIES as not dataset
    if use_series:
        if SERIES is None:
            raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
    # Check user input
    if dataset is not None:
        if input_form == 'series':
            if isinstance(dataset, pd.Series):
                print('Get input pd.Series named:', str(dataset.name))
                input_dataset = dataset.copy(deep=True)
            else:
                raise ValueError('The inputting series must be pd.Seriesrather than %s.' % type(dataset))
        elif input_form == 'df':
            if isinstance(dataset, pd.DataFrame):
                print('Get input pd.DataFrame.')
                tmp_sum = None
                if 'sum' in dataset.columns:
                    tmp_sum = dataset['sum']
                    dataset = dataset.drop('sum', axis=1, inplace=False)
                if 'co-imf0' in dataset.columns:
                    col_name = 'co-imf'
                else:
                    col_name = 'imf'
                dataset.columns = [col_name + str(i) for i in
                                   range(len(dataset.columns))]  # change column names to imf0,imf1,...
                if tmp_sum is not None:  dataset['sum'] = tmp_sum
                input_dataset = dataset.copy(deep=True)
            else:
                raise ValueError('The inputting df must be pd.DataFrame rather than %s.' % type(dataset))
        else:
            raise ValueError('Something wrong happen in module %s.' % __name__)
        file_name = 'test_'
    else:  # Check default dataset and load
        if input_form == 'series':  # Check SERIES
            if not isinstance(SERIES, pd.Series):
                raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
            else:
                input_dataset = SERIES.copy(deep=True)
        elif input_form == 'df':
            check_vars()
            data_path = PATH + MODE + FORM + '_data.csv'
            if not os.path.exists(data_path):
                raise ImportError('Dataset %s does not exist in ' % (data_path) + PATH)
            else:
                input_dataset = pd.read_csv(data_path, header=0, index_col=0)

    # other warnings
    if METHOD == 0 and uni_nor:
        print(
            'Attention!!! METHOD = 0 means no using the unified normalization method. Declare METHOD by declare_uni_method(method=METHOD)')

    return input_dataset, file_name


# Declare LSTM model variables
# -------------------------------
# The units of LSTM layers and 3 LSTM layers will set to 4*CELLS, 2*CELLS, CELLS.
CELLS = 32
# Dropout rate of 3 Dropout layers
DROPOUT = 0.2
# Adam optimizer loss such as 'mse','mae','mape','hinge' refer to https://keras.io/zh/losses/
OPTIMIZER_LOSS = 'mse'
# LSTM training batch_size for parallel computing, suggest 10-100
BATCH_SIZE = 200
# Proportion of validation set to training set, suggest 0-0.2
VALIDATION_SPLIT = 0.1
# Report of the training process, 0 not displayed, 1 detailed, 2 rough
VERBOSE = 0
# In the training process, whether to randomly disorder the training set
SHUFFLE = True


# Declare LSTM variables
def declare_LSTM_vars(cells=CELLS, dropout=DROPOUT, optimizer_loss=OPTIMIZER_LOSS, batch_size=BATCH_SIZE,
                      validation_split=VALIDATION_SPLIT, verbose=VERBOSE, shuffle=SHUFFLE):
    print('##################################')
    print('LSTM Model Variables')
    print('##################################')
    PATIENCE
    # Changepatience=
    global CELLS, DROPOUT, OPTIMIZER_LOSS, BATCH_SIZE, VALIDATION_SPLIT, VERBOSE, SHUFFLE
    CELLS, DROPOUT, OPTIMIZER_LOSS = cells, dropout, optimizer_loss
    BATCH_SIZE, VALIDATION_SPLIT, VERBOSE, SHUFFLE = batch_size, validation_split, verbose, shuffle

    # Check
    if not (type(CELLS) == int and CELLS > 0): raise TypeError('CELLS should a positive integer.')
    if not (type(DROPOUT) == float and DROPOUT > 0 and DROPOUT < 1): raise TypeError(
        'DROPOUT should a number between 0 and 1.')
    if not (type(BATCH_SIZE) == int and BATCH_SIZE > 0):
        raise TypeError('BATCH_SIZE should be a positive integer.')
    if not (type(VALIDATION_SPLIT) == float and VALIDATION_SPLIT > 0 and VALIDATION_SPLIT < 1):
        raise TypeError('VALIDATION_SPLIT should be a number best between 0.1 and 0.4.')
    if VERBOSE not in [0, 1, 2]:
        raise TypeError('VERBOSE should be 0, 1, or 2. The detail level of the training message')
    if type(SHUFFLE) != bool:
        raise TypeError('SHUFFLE should be True or False.')

    # Show
    print('CELLS:' + str(CELLS))
    print('DROPOUT:' + str(DROPOUT))
    print('OPTIMIZER_LOSS:' + str(OPTIMIZER_LOSS))
    print('BATCH_SIZE:' + str(BATCH_SIZE))
    print('VALIDATION_SPLIT:' + str(VALIDATION_SPLIT))
    print('VERBOSE:' + str(VERBOSE))
    print('SHUFFLE:' + str(SHUFFLE))


# Define the Keras model by model = Sequential() with input shape [DATE_BACK,the number of features]
LSTM_MODEL = None


# Change Kreas model
def declare_LSTM_MODEL(model=LSTM_MODEL):
    print("LSTM_MODEL has changed to be %s and start your forecast." % model)
    global LSTM_MODEL
    
    LSTM_MODEL = model


# LSTM model example
def LSTM_example():
    print('Please input a Keras model with input_shape = (DATE_BACK, the number of features)')
    print('##################################')
    print("model = Sequential()")
    print("model.add(LSTM(100, input_shape=(30, 1), activation='tanh'))")
    print("model.add(Dropout(0.5))")
    print("model.add(Dense(1,activation='tanh'))")
    print("model.compile(loss='mse', optimizer='adam')")
    print("cl.declare_LSTM_MODEL(model=model)")


# Build LSTM model
def LSTM_model(shape, shape2=None):
    if LSTM_MODEL is None:
        model = Sequential()
        model.add(LSTM(CELLS * 4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS * 2, activation='tanh', return_sequences=False))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'Adjust_IMF':
        model = Sequential()
        model.add(LSTM(CELLS * 4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS * 2, activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS, activation='tanh', return_sequences=False))
        model.add(Dropout(DROPOUT))
        model.add(Dense(shape2[2], activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'GRU':
        model = Sequential()
        model.add(GRU(CELLS * 4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(GRU(CELLS * 2, activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(GRU(CELLS, activation='tanh', return_sequences=False))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'DNN':
        model = Sequential()
        model.add(Dense(CELLS * 4, input_shape=(shape[1], shape[2]), activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(CELLS * 2, activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(CELLS, activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'BPNN':
        model = Sequential()
        model.add(Dense(CELLS * 4, input_shape=(shape[1], shape[2]), activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    else:
        return LSTM_MODEL


"""
# TCN
model = Sequential()
model.add(TCN(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh'))
model.add(Dropout(DROPOUT))
model.add(Dense(1,activation='tanh'))
model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
return model
"""
# Other variables
# -------------------------------
# Method for unified normalization only 0,1,2,3
METHOD = 0


# declare Method for unified normalization
def declare_uni_method(method=None):
    if method not in [0, 1, 2, 3]: raise TypeError('METHOD should be 0,1,2,3.')
    global METHOD
    METHOD = method
    print('Unified normalization method (%d) is start using.' % method)


# 3.Decomposition, Sample entropy, Re-decomposition, and Integration
# ==============================================================================================

# EMD decomposition
# -------------------------------
# Decompose adaptively and plot function
# Residue is named the last IMF
# Declare MODE by declare_vars first
def emd_decom(series=None, trials=10, max_imf= None, re_decom=False, re_imf=0, draw=True , save_name = ''):
    # Check input
    dataset, file_name = check_dataset(series, input_form='series')  # include check_vars()
    series = dataset.values
    MAX_IMF = max_imf
    # Initialization
    print('%s decomposition is running.' % str.upper(MODE))
    if MODE == 'emd':
        decom = EMD()
        # decom.FIXE_H = 5
        # decom.nbsym = 10
        # decom.spline_kind = "cubic"

    elif MODE == 'eemd':
        decom = EEMD()
    elif MODE == 'ceemdan':
        decom = CEEMDAN()
    else:
        raise ValueError('MODE must be emd, eemd, ceemdan when EMD decomposing.')

    # Decompose
    decom.trials = trials  # Number of the white noise input
    if MAX_IMF is not None:
        imfs_emd = decom(series,max_imf = MAX_IMF)
    else:
        imfs_emd = decom(series)
    imfs_num = np.shape(imfs_emd)[0]

    if draw:
        # Plot original data
        series_index = range(len(series))
        fig = plt.figure(figsize=(16, 2 * imfs_num))
        plt.subplot(1 + imfs_num, 1, 1)
        plt.plot(series_index, series, color='#0070C0')  # F27F19 orange #0070C0 blue
        plt.ylabel('Original data')

        # Plot IMFs
        for i in range(imfs_num):
            plt.subplot(1 + imfs_num, 1, 2 + i)
            plt.plot(series_index, imfs_emd[i, :], color='#F27F19')
            plt.ylabel(str.upper(MODE) + '-IMF' + str(i))

        # Save figure
        fig.align_labels()
        plt.tight_layout()
        if file_name == '':
            if (re_decom == False):
                plt.savefig(FIGURE_PATH + file_name + str.upper(MODE) + ' Result.svg', bbox_inches='tight')
            else:
                plt.savefig(FIGURE_PATH + 'IMF' + str(re_imf) + ' ' + str.upper(MODE) + ' Re-decomposition Result.svg',
                            bbox_inches='tight')
        plt.show()

        #绘制频谱图
        N= len(series)
        K = imfs_num
        Fs = 1
        Ts = 1.0/Fs
        t = np.arange(N)
        small_k = np.arange(N)
        T = N/Fs
        frq = small_k/T
        frq1 = frq[range(int(N/2))]
        
        plt.figure(figsize=(16, 2 * imfs_num))
        #plt.figure(figsize=(10, 8))
        for i in range(K):
            
            plt.subplot(K + 1, 1, 1)
            data_f = abs(np.fft.fft(series)) / N
            data_f1 = data_f[range(int(N / 2))]
            plt.plot(frq1, data_f1)
            plt.title("outer")
            plt.subplot(K+1, 1, i + 2)
            data_f2 = abs(np.fft.fft(imfs_emd[i, :])) / N
            data_f3 = data_f2[range(int(N / 2))]
            plt.plot(frq1, data_f3, 'red')
            plt.xlabel('pinlv(hz)')
            plt.ylabel('u{}'.format(i + 1))
        plt.tight_layout()
        plt.show()



        "如果将横坐标转换成周期来表示"
                # 计算周期向量，忽略零频率
        periods = np.where(frq1 != 0, 1 / frq1, np.inf)

        # 绘制频谱图（示例代码）
        plt.figure(figsize=(16, 2 * imfs_num))

        # 原始信号的频谱
        plt.subplot(K + 1, 1, 1)
        plt.plot(periods, data_f1)
        plt.xlabel('Period (days)')
        plt.ylabel('Amplitude')
        plt.title("Period Spectrum of Original Series")

        # 每个 IMF 的频谱
        for i in range(K):
            plt.subplot(K + 1, 1, i + 2)
            plt.plot(periods, data_f3, color='red')
            plt.xlabel('Period (days)')
            plt.ylabel(f'IMF {i + 1}')

        plt.tight_layout()
        plt.show()

    # Save data
    imfs_df = pd.DataFrame(imfs_emd.T)
    imfs_df.columns = ['imf' + str(i) for i in range(imfs_num)]

    # pd.DataFrame.to_csv(imfs_df, PATH.strip('\\') + '/decompose/' + MODE + save_name + '_data.csv')
    # print(str.upper(MODE) + ' finished, check the dataset: ', PATH.strip('\\') + 'decompose/' + MODE + save_name + '_data.csv')

    return imfs_df  # pd.DataFrame


# Sample entropy
# -------------------------------
# You can also enter the imfs_df directly
def sample_entropy(imfs_df=None):  # imfs_df is pd.DataFrame
    df_emd, file_name = check_dataset(imfs_df, input_form='df')  # include check_vars()
    if file_name == '':
        file_name = str.upper(MODE + FORM)
    else:
        file_name = 'a Test'
    print('Sample entropy of %s is running.' % file_name)

    # Calculate sample entropy with m=1,2 and r=0.1,0.2
    imfs = df_emd.T.values
    sampen = []
    for i in imfs:
        for j in (0.1, 0.2):
            sample_entropy = sampen2(list(i), mm=2, r=j, normalize=True)   #奇数 第1 3 5..行放的是r = 0.1的熵
            sampen.append(sample_entropy)

    # Output
    entropy_r1m1, entropy_r1m2, entropy_r2m1, entropy_r2m2 = [], [], [], []
    for i in range(len(sampen)):
        if (i % 2) == 0:  # r=0.1
            entropy_r1m1.append(sampen[i][1][1])  # m=1       #取出熵值时同样依据奇偶，来拿出相应熵
            entropy_r1m2.append(sampen[i][2][1])  # m=2
        else:  # r=0.2
            entropy_r2m1.append(sampen[i][1][1])  # m=1
            entropy_r2m2.append(sampen[i][2][1])  # m=2

    # Plot
    fig = plt.figure()
    x = list(range(0, len(imfs), 1))
    plt.plot(x, entropy_r1m1, 'k:H', label='m=1 r=0.1')
    plt.plot(x, entropy_r2m1, 'b:D', label='m=1 r=0.2')
    plt.plot(x, entropy_r1m2, 'c:s', label='m=2 r=0.1')
    plt.plot(x, entropy_r2m2, 'm:h', label='m=2 r=0.2')   
    plt.xlabel('IMFs')
    plt.ylabel('Sample Entropy')
    plt.legend()
    if file_name == '': fig.savefig(FIGURE_PATH + 'Sample Entropy of %s IMFs.svg' % (file_name), bbox_inches='tight')
    plt.show()

    return entropy_r1m1, entropy_r2m1, entropy_r1m2, entropy_r2m2

#模拟专家经验判断，定义一个自动根据样本熵结果来进行IMFs分组判别操作的函数



def once_auto_group(list=None):   #对输入的列表数据自动进行一次分组，默认列表的元素中均是整数
    
    once_diff = []
    for i in range(len(list)):
        for j in range(len(list)):
            if type(list[i]) == list:
                print('==================出现错误!该输入变量中不能含有list=========================')
            if i != j :
                temp_diff = abs(list[i]-list[j])
                once_diff.append(copy.deepcopy(temp_diff))
    
    index = once_diff.index(min(once_diff))   #返回列表中第一个出现的最小值的索引位置
    lenth = len(list)
    site = index + 1  #由于索引位置是从0开始 +1更易理解元素位数

    if  site%(lenth-1) == 0: #如果余数为0
        x = site//(lenth-1) 
        if x == lenth:
            temp_y = -2
            y = lenth - 1
        else:
            temp_y = -1 
            y = lenth

    else: # 如果余数不为0，则不涉及最后一位数字的边界问题
        x = site//(lenth-1) + 1          #找出是第x个元素
        temp_y = site%(lenth-1)               #求余

        if temp_y < x:                   
            y = temp_y

        else:                             #去除列表中自己所处位置
            y = temp_y + 1
 



    #得到这一次所有元素计算差值时，最小数值对应的两个位置  即x,y位置的元素，两两之间距离最小
    return x, y, list[x-1], list[y-1]               #(x,y)指的是第X和第Y个元素，不是索引，位置从1开始计数


def retry_auto_group(Origianl_se=None, retry_se=None):  #如果在完成了一个完整的分组过程后，我们认为有的分组中元素过多，可以再对该分组进行一次细分


    #retry_list为：元素过多的分组list
    #retry_se为：元素过多的分组list中元素对应的平均熵值list
    
    temp_list = copy.deepcopy(retry_se)
    #进行第一次分组，首先判断所有平均样本熵值是否有相等的
    x, y, num_x, num_y = once_auto_group(list=temp_list)
    save_dict = {} 
    if num_x == num_y:
        print('================================重分组过程中，所有平均样本熵值中存在数值相等的情况=========================')
        #由于样本熵的计算数值取到小数点后16位，可以认为出现数值相等的事件为小概率事件，基本不会发生，这里代码中不再设置该事件发生的执行代码，只报告该事件的发生
        pass
        
        
    else:
      
        print('================================重分组过程中，所有平均样本熵值都两两不相等=========================')

        
        
        for i in range(100):
            print('第%s次循环分组中'%(i+1))
            if i == 0:  #第一次进行IMFs分组

                index_x,index_y = x-1,y-1  #他们的索引位置是各自减一
                #将第x,y个IMFs归为一组，并放置他们的欧式距离中心进temp_list
                new_temp_list = [x for x in temp_list if x not in [num_x, num_y] ]  #在temp_list中删去这两个已分组的数字
                new_num = (num_x +  num_y)/2
                save_dict[new_num] = [num_x, num_y]  #将这两个数字的平均数作为字典的KEY
                new_temp_list.append(copy.deepcopy(new_num))   #将这两个数字的平均数加入list中继续计算欧式距离

                
            else:
                
                #如果再分组过程中，元素大小=3，则另外两个元素自动分为一组，同样停止迭代
                for item in save_dict:
                    if len(save_dict[item]) == 3:
                        remove_key = np.average(save_dict[item])
                        if remove_key in new_temp_list:
                            new_temp_list.remove(remove_key)         #移除该item的key,即不再参与分组计算
                            #如果new_temp_list移除列表后只剩下一个元素，那么这个元素自己单独成为一组


                #迭代停止的条件为：save_dict中的元素个数是否等于最初输入的list长度
                sum = 0
                for item in save_dict:
                    sum = copy.deepcopy(sum) + len(save_dict[item])
                
                if sum == len(retry_se) :  #在retry中循环次数不需要减一
                    break
                else:
                    if len(new_temp_list) == 1:
                        save_dict[new_temp_list[0]] = new_temp_list
                        if sum+1 == len(retry_se): break
                print('本次重新分组的retry_se为:',retry_se)        
                print('目前save_dict为:',save_dict)
                print('目前new_temp_list中元素为:',new_temp_list)

                #list进行过一次分组后，继续分组
                x, y, num_x, num_y = once_auto_group(list=new_temp_list)
                new_temp_list = [x for x in new_temp_list if x not in [num_x, num_y] ] 
                
                #两个分组之间的欧式距离中心也可以继续计算是否为距离最近
                
                #首先判断 num_x，num_y是否是已分组数据，是的话取出他们的已分组数据
                if num_x in save_dict: num_x = save_dict[num_x]
                if num_y in save_dict: num_y = save_dict[num_y]

                #如果两者经过在字典查询后，仍然都是数字，说明都没经过分组，那么将他们划分到一组
                if type(num_x) != list and type(num_y) != list:
                    new_num = (num_x +  num_y)/2
                    save_dict[new_num] = [num_x, num_y] 
                    new_temp_list.append(copy.deepcopy(new_num))
                #如果两者经过字典查询后，有一个已分组，那么将未分组的num_x纳入其中
                if type(num_x) != list and type(num_y) == list:

                    new_list = copy.deepcopy(num_y)         #复制一份旧的list 保留num_y以计算旧的KEY值
                    new_list.append(copy.deepcopy(num_x))     #形成新的分组list
                    old_key = np.average(num_y)
                    new_num = np.average(new_list)               #计算新的平均数

                    save_dict.pop(old_key)                          #删去旧的分组
                    save_dict[new_num] = new_list                   #保留新的分组
                    new_temp_list.append(copy.deepcopy(new_num))    #放入新的列表均值继续计算


                if type(num_x) == list and type(num_y) != list:

                    new_list = copy.deepcopy(num_x)      #复制一份旧的list 保留num_x以计算旧的KEY值
                    new_list.append(copy.deepcopy(num_y))  #形成新的分组list
                    old_key = np.average(num_x)
                    new_num = np.average(new_list)               #计算新的平均数

                    save_dict.pop(old_key)                          #删去旧的分组
                    save_dict[new_num] = new_list                   #保留新的分组
                    new_temp_list.append(copy.deepcopy(new_num))    #放入新的列表均值继续计算


                 #如果两者经过在字典查询后，发现都是list，说明都经过分组，也可以将他们合为一组
                if type(num_x) == list and type(num_y) == list:
                    save_dict.pop(np.average(num_x))  
                    save_dict.pop(np.average(num_y))  #删除旧的分组

                    new_list = num_x + num_y          #生成新的分组  列表相加会自动进行拼接

                    new_num = np.average(new_list)    #生成新的KEY
                    save_dict[new_num] = new_list     #保存新的分组
                    new_temp_list.append(copy.deepcopy(new_num))       #将新的均值放入继续计算
            print('目前生成的save_dict如下所示:', save_dict)


    #把字段中存储的样本平均熵数值转换为IMFs的序号
    group_all = []
    for item in save_dict:
        temp_list = save_dict[item]
        group = []
        for num in temp_list:
            site = Origianl_se.index(num)
            group.append(copy.deepcopy(site))
        group_all.append(copy.deepcopy(group))


    return group_all


#对输入的IMFs数据自动计算样本熵，并自动对样本熵进行分组，直到所有的IMFs都已经被分组，并输出分组情况
def Auto_Se_Group(imfs_df=None):

    entropy_r1m1, entropy_r2m1, entropy_r1m2, entropy_r2m2 = sample_entropy(imfs_df=imfs_df)

    #原理是，首先对四种情况下IMFs的熵值取平均作为IMFs的平均样本熵值
    #那么问题就转换为对一个列表中的数值进行分组的问题，采取的原则是，在每一次遍历中，总是将欧式距离最小的两个数字判定为一组
    #分为一组后，重新计算该组的欧式距离中心，并将该数值作为下一次遍历的依据

    ave_se = [(a+b+c+d)/4 for a,b,c,d in zip(entropy_r1m1,entropy_r2m1,entropy_r1m2,entropy_r2m2)]
    
    #我们总是把最后一个IMF序列即残差，单独作为一组，所以首先把他摘除出去

    temp_list = copy.deepcopy(ave_se)
    temp_list.pop(-1)
    #进行第一次分组，首先判断所有平均样本熵值是否有相等的
    x, y, num_x, num_y = once_auto_group(list=temp_list)
    save_dict = {} 
    if num_x == num_y:
        print('================================所有平均样本熵值中存在数值相等的情况=========================')
        #由于样本熵的计算数值取到小数点后16位，可以认为出现数值相等的事件为小概率事件，基本不会发生，这里代码中不再设置该事件发生的执行代码，只报告该事件的发生
        pass
        
        
    else:
      
        print('================================所有平均样本熵值都两两不相等=========================')

        
        
        for i in range(100):
            print('第%s次循环分组中'%(i+1))
            if i == 0:  #第一次进行IMFs分组

                index_x,index_y = x-1,y-1  #他们的索引位置是各自减一
                #将第x,y个IMFs归为一组，并放置他们的欧式距离中心进temp_list
                new_temp_list = [x for x in temp_list if x not in [num_x, num_y] ]  #在temp_list中删去这两个已分组的数字
                new_num = (num_x +  num_y)/2
                save_dict[new_num] = [num_x, num_y]  #将这两个数字的平均数作为字典的KEY
                new_temp_list.append(copy.deepcopy(new_num))   #将这两个数字的平均数加入list中继续计算欧式距离

                
            else:
                
                #迭代停止的条件为：save_dict中的元素个数是否等于最初输入的list长度
                sum = 0
                for item in save_dict:
                    sum = copy.deepcopy(sum) + len(save_dict[item])
                
                if sum == len(ave_se) - 1:  #考虑到实际分组中去除了Residual 所以循环次数应该减一
                    break
                #list进行过一次分组后，继续分组
                x, y, num_x, num_y = once_auto_group(list=new_temp_list)
                new_temp_list = [x for x in new_temp_list if x not in [num_x, num_y] ] 
                
                #两个分组之间的欧式距离中心也可以继续计算是否为距离最近
                
                #首先判断 num_x，num_y是否是已分组数据，是的话取出他们的已分组数据
                if num_x in save_dict: num_x = save_dict[num_x]
                if num_y in save_dict: num_y = save_dict[num_y]

                #如果两者经过在字典查询后，仍然都是数字，说明都没经过分组，那么将他们划分到一组
                if type(num_x) != list and type(num_y) != list:
                    new_num = (num_x +  num_y)/2
                    save_dict[new_num] = [num_x, num_y] 
                    new_temp_list.append(copy.deepcopy(new_num))
                #如果两者经过字典查询后，有一个已分组，那么将未分组的num_x纳入其中
                if type(num_x) != list and type(num_y) == list:

                    new_list = copy.deepcopy(num_y)         #复制一份旧的list 保留num_y以计算旧的KEY值
                    new_list.append(copy.deepcopy(num_x))     #形成新的分组list
                    old_key = np.average(num_y)
                    new_num = np.average(new_list)               #计算新的平均数

                    save_dict.pop(old_key)                          #删去旧的分组
                    save_dict[new_num] = new_list                   #保留新的分组
                    new_temp_list.append(copy.deepcopy(new_num))    #放入新的列表均值继续计算


                if type(num_x) == list and type(num_y) != list:

                    new_list = copy.deepcopy(num_x)      #复制一份旧的list 保留num_x以计算旧的KEY值
                    new_list.append(copy.deepcopy(num_y))  #形成新的分组list
                    old_key = np.average(num_x)
                    new_num = np.average(new_list)               #计算新的平均数

                    save_dict.pop(old_key)                          #删去旧的分组
                    save_dict[new_num] = new_list                   #保留新的分组
                    new_temp_list.append(copy.deepcopy(new_num))    #放入新的列表均值继续计算


                 #如果两者经过在字典查询后，发现都是list，说明都经过分组，也可以将他们合为一组
                if type(num_x) == list and type(num_y) == list:
                    save_dict.pop(np.average(num_x))  
                    save_dict.pop(np.average(num_y))  #删除旧的分组

                    new_list = num_x + num_y          #生成新的分组  列表相加会自动进行拼接

                    new_num = np.average(new_list)    #生成新的KEY
                    save_dict[new_num] = new_list     #保存新的分组
                    new_temp_list.append(copy.deepcopy(new_num))       #将新的均值放入继续计算
            print('目前生成的save_dict如下所示:', save_dict)
    #把字段中存储的样本平均熵数值转换为IMFs的序号
    group_all = []
    for item in save_dict:
        temp_list = save_dict[item]
        group = []
        for num in temp_list:
            site = ave_se.index(num)
            group.append(copy.deepcopy(site))
        group_all.append(copy.deepcopy(group))


    #由于之前的分组里，我们将Residual摘除出去，现在把他作为单独一组放回到group_all
    #Resdiual的位置信息等于len(entropy_r1m1) - 1

    Resdiual = len(entropy_r1m1) - 1
    group_all.append([Resdiual])

    #如果group_all中某个分组的元素>=4，则认为分组不够细，那么继续进行一次retry_auto_group
    for item in group_all:
        if len(item) >=4:
            print('重新对元素过多的分组进行细分retry_auto_group,该分组如下：', item)
            retry_se = [x for index, x in enumerate(ave_se) if index in item]
            new_group = retry_auto_group(Origianl_se=ave_se, retry_se=retry_se)   #对元素过多的list产生新的分组
            group_all.remove(item)          #在group_all中删去旧的分组
            for new_item in new_group:
                group_all.append(copy.deepcopy(new_item))
        
    #group中存放的是所有IMFs的分组信息，里面的数值代表IMFs的编号顺序
    print('group_all如下所示:',group_all)
    print('ave_se如下所示:',ave_se)

    #根据group_all和IMFs，输出重构后的R-IMFS
    
    # for item in range(group_all):
    #     for i in item:
    #         pass

    return group_all, ave_se


# 4.LSTM Model Functions
# ==============================================================================================

# Model evaluation function
# -------------------------------
def evl(y_test, y_pred, scale='0 to 1'):  # MSE and MAE are different on different scal    y_test, y_pred = np.array(y_test).ravel(), np.array(y_pred).ravel()
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)  
    sum =0
    for i in range(len(y_test)-1):
        temp = 0
        if (y_test[i+1] - y_test[i]) * (y_pred[i+1] - y_test[i]) >= 0 :
            temp = 1
        else:
            temp = 0
        sum = copy.deepcopy(sum) + copy.deepcopy(temp)
    Dstat = sum / len(y_test)
    
    print('##################################')
    print('Model Evaluation with scale of', scale)
    print('##################################')
    print('R2:', round(r2,3))
    print('RMSE:', round(rmse,3))
    print('MAE:', round(mae,3))
    print("MAPE:", round(mape,3))  # MAPE before normalization may error beacause of negative values
    print("Dstat:", round(Dstat,3))
    return [round(r2,3), round(rmse,3), round(mae,3), round(mape,3),round(Dstat,3)]


# DATE_BACK functions for inputting sets
# -------------------------------
# IMPORTANT!!! it may cause some error when the input format is wrong.
# Method here is used to determine the Unified normalization, use declare_uni_method(method=METHOD) to declare.
def create_dateback(df, uni=False, ahead=1):
    # Normalize for DataFrame
    if uni and METHOD != 0 and ahead == 1:  # Unified normalization
        # Check input and load dataset
        if SERIES is None: raise ValueError(
            'SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
        if MODE not in ['emd', 'eemd', 'ceemdan']: raise ValueError(
            'MODE must be emd, eemd, ceemdan if you want to try unified normalization method.')
        if not (os.path.exists(PATH + MODE + '_data.csv')): raise ImportError(
            'Dataset %s does not exist in ' % (PATH + MODE + '_data.csv'), PATH)

        # Load data
        df_emd = pd.read_csv(PATH + MODE + '_data.csv', header=0, index_col=0)
        # Method (1)
        print('##################################')
        if METHOD == 1:
            scalar, min0 = SERIES.max() - SERIES.min(), 0
            print('Unified normalization Method (1):')
        # Method (2)
        elif METHOD == 2:
            scalar, min0 = df_emd.max().max() - df_emd.min().min(), df_emd.min().min()
            print('Unified normalization Method (2):')
        # Method (3)
        elif METHOD == 3:
            scalar, min0 = SERIES.max() - df_emd.min().min(), df_emd.min().min()
            print('Unified normalization Method (3):')

        # Normalize
        df = (df - min0) / scalar
        scalarY = {'scalar': scalar, 'min': min0}
        print(df)
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            trainY = np.array(df['sum']).reshape(-1, 1)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            trainX = trainY
    else:
        # Normalize without unifying
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            scalarX = MinMaxScaler(feature_range=(0, 1))  # sklearn normalize
            trainX = scalarX.fit_transform(trainX)
            trainY = np.array(df['sum']).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0, 1))  # sklearn normalize
            trainY = scalarY.fit_transform(trainY)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0, 1))  # sklearn normalize
            trainY = scalarY.fit_transform(trainY)
            trainX = trainY

    # Create dateback
    dataX, dataY = [], []
    ahead = ahead - 1
    for i in range(len(trainY) - DATE_BACK - ahead):
        dataX.append(np.array(trainX[i:(i + DATE_BACK)]))
        dataY.append(np.array(trainY[i + DATE_BACK + ahead]))
    return np.array(dataX), np.array(dataY), scalarY, np.array(trainX[-DATE_BACK:])


# Plot original data and forecasting data
def plot_all(lstm_type, pred_ans):
    # Check and Change
    if not isinstance(SERIES, pd.Series):
        raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
    pred_ans = pred_ans.ravel()
    series_pred = SERIES.copy(deep=True)  # copy original data
    for i in range(PERIODS):
        series_pred[-i - 1] = pred_ans[-i - 1]

    # Plot
    fig = plt.figure(figsize=(10, 4))
    SERIES[-PERIODS * 3:].plot(label='Original data', color='#0070C0')  # F27F19 orange #0070C0 blue
    series_pred[-PERIODS:].plot(label='Forecasting data', color='#F27F19')
    plt.xlabel('')
    plt.title(lstm_type + ' LSTM forecasting results')
    plt.legend()
    plt.savefig(FIGURE_PATH + lstm_type + ' LSTM forecasting results.svg', bbox_inches='tight')
    plt.show()
    return


# Declare LSTM forecasting function
# Have declared LSTM model variables at Section 0 before
# -------------------------------
def LSTM_pred(data=None, draw=True, uni=False, show_model=True, train_set=None, next_pred=False, ahead=1):
    # Divide the training and test set
    if train_set is None:
        trainX, trainY, scalarY, next_trainX = create_dateback(data, uni=uni, ahead=ahead)
    else:
        trainX, trainY, scalarY, next_trainX = train_set[0], train_set[1], train_set[2], train_set[3]
    if uni == True and next_pred == True: raise ValueError('Next pred does not support unified normalization.')

    if PERIODS == 0:
        train_X = trainX
        y_train = trainY
    else:
        x_train, x_test = trainX[:-PERIODS], trainX[-PERIODS:]
        y_train, y_test = trainY[:-PERIODS], trainY[-PERIODS:]
        # Convert to tensor
        train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Build and train the model
    # print('trainX:\n',train_X[-1:])
    print('\nInput Shape: (%d,%d)\n' % (train_X.shape[1], train_X.shape[2]))
    model = LSTM_model(train_X.shape)
    if show_model: model.summary()  # The summary of layers and parameters
    EarlyStop = EarlyStopping(monitor='val_loss', patience=5 * PATIENCE, verbose=VERBOSE,
                              mode='auto')  # realy stop at small learning rate
    Reduce = ReduceLROnPlateau(monitor='val_loss', patience=PATIENCE, verbose=VERBOSE,
                               mode='auto')  # Adaptive learning rate
    history = model.fit(train_X, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                        verbose=VERBOSE, shuffle=SHUFFLE, callbacks=[EarlyStop, Reduce])

    # Plot the model structure
    # plot_model(model,to_file=FIGURE_PATH+'model.png',show_shapes=True)

    # Predict
    if PERIODS != 0:
        pred_test = model.predict(test_X)
        # Evaluate model with scale 0 to 1
        evl(y_test, pred_test)
    else:
        pred_test = np.array([])

    if next_pred:  # predict tomorrow not in test set
        next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
        pred_test = np.append(pred_test, next_ans)
    pred_test = pred_test.ravel().reshape(-1, 1)

    # De-normalize
    # IMPORTANT!!! It may produce some negative data impact evaluating
    if isinstance(scalarY, MinMaxScaler):
        test_pred = scalarY.inverse_transform(pred_test)
        if PERIODS != 0: test_y = scalarY.inverse_transform(y_test)
    else:
        test_pred = pred_test * scalarY['scalar'] + scalarY['min']
        if PERIODS != 0: test_y = y_test * scalarY['scalar'] + scalarY['min']

    # Plot
    if draw and PERIODS != 0:
        # determing the output name of figures
        fig_name = ''
        if isinstance(data, pd.Series):
            if str(data.name) == 'None':
                fig_name = 'Series'
            else:
                fig_name = str(data.name)
        else:
            fig_name = 'DataFrame'

        # Plot the loss figure
        fig = plt.figure(figsize=(5, 2))
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.title(fig_name + ' LSTM loss chart')
        plt.savefig(FIGURE_PATH + fig_name + ' LSTM loss chart.svg', bbox_inches='tight')
        plt.show()

        # Plot observation figures
        fig = plt.figure(figsize=(5, 2))
        plt.plot(test_y)
        plt.plot(test_pred)
        plt.title(fig_name + ' LSTM forecasting result')
        plt.savefig(FIGURE_PATH + fig_name + ' LSTM forecasting result.svg', bbox_inches='tight')
        plt.show()

    return test_pred

# 5.CEEMDAN-LSTM Forecasting Functions
# Please use cl.declare_vars() to determine variables.
# ==============================================================================================

# Single LSTM Forecasting without CEEMDAN
# -------------------------------
# It uses LSTM directly for prediction wiht input_shape=[DATE_BACK,1]
def Single_LSTM(series=None, draw=True, uni=False, show_model=True, ahead=1):
    print('==============================================================================================')
    print('This is Single LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input series and load dataset

    #利用训练集特征对训练集和测试集进行标准化
    dataX, dataY, (max, min) = Normalize(series_train= series[:-PERIODS], series_test= series[-PERIODS:])
    dataX.extend(dataY)
    dataX = pd.Series(dataX)
    start = time.time()
    #生成训练、测试样本对
    trainX,trainY,next_trainX = Roll_create_dateback(dataX,ahead=ahead)

    x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
    y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]
    
    train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    next_trainX = next_trainX.reshape(1, next_trainX.shape[0], next_trainX.shape[1])
    
    model = LSTM_model(trainX.shape)
    # The summary of layers and parameters
    model.summary() 

    history = model.fit(train_X, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                verbose=VERBOSE, shuffle=SHUFFLE)
    
    pred_test = model.predict(test_X)
    # Evaluate model with scale of inverse                #0 to 1

    y_test = y_test*(max -min) + min
    pred_test = pred_test*(max - min) + min
    print('当前%s模型预测结果如下:'%LSTM_MODEL)
    evl_indicators = evl(y_test, pred_test)

    #得到所有预测结果

    end = time.time()
    print('Running time: %.3fs' % (end - start))
    df_evl = pd.DataFrame(pred_test)
    #保存结果
    pd.DataFrame.to_csv(df_evl, LOG_PATH + SERIES.name  + 'single_LSTM_log.csv', index=False, header=0, mode='a')  # log record
    print('Single LSTM Forecasting finished, check the logs', LOG_PATH +  'single_LSTM_log.csv')

    return pred_test,evl_indicators

# =======================没有标准化操作的Single_LSTM========================================
def Single_LSTM_Nonormallize(series=None, draw=True, uni=False, show_model=True, ahead=1):
    print('==============================================================================================')
    print('This is Single LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input series and load dataset

    #利用训练集特征对训练集和测试集进行标准化
    dataX = series[:-PERIODS]
    dataY = series[-PERIODS:]
    if type(series) == 'list':
        dataX.extend(dataY)
        dataX = pd.Series(dataX)
    else:
        dataX.append(dataY)
    start = time.time()
    #生成训练、测试样本对
    trainX,trainY,next_trainX = Roll_create_dateback(dataX,DATE_BACK=DATE_BACK,ahead=ahead)

    x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
    y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]
    
    train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    next_trainX = next_trainX.reshape(1, next_trainX.shape[0], next_trainX.shape[1])
    
    model = LSTM_model(trainX.shape)
    # The summary of layers and parameters
    model.summary() 
    print('=============开始Fiting=========================')
    history = model.fit(train_X, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                verbose=VERBOSE, shuffle=SHUFFLE)
    
    pred_test = model.predict(test_X)
    # Evaluate model with scale 0 to 1
    print('当前%s模型预测结果如下:'%LSTM_MODEL)
    evl(y_test, pred_test)

    #得到所有预测结果
    
    test_pred = list( pred_test.ravel() )
    test_y = list(y_test)

    end = time.time()
    print('Running time: %.3fs' % (end - start))
    df_evl = pd.DataFrame(test_pred)
    #保存结果
    pd.DataFrame.to_csv(df_evl, LOG_PATH + SERIES.name  + 'single_LSTM_log.csv', index=False, header=0, mode='a')  # log record
    print('Single LSTM Forecasting finished, check the logs', LOG_PATH +  'single_LSTM_log.csv')

    return df_evl    
#========================================滚动分解================================================================

def Rolldecom(data, PERIODS):
    import copy  #使用copy函数，避免append时由于指向地址（而不是真正将值赋予列表）而导致的当对象改变时，列表也跟着改变
    trainX, trainY = data[:-PERIODS], data[-PERIODS:]

    trainX = list(trainX)
    trainY = list(trainY)
    trainX_all = []
    trainY_all = []
    trainY0 = copy.deepcopy(trainY)

    trainX_all.append(copy.deepcopy(trainX))
    trainY_all.append(copy.deepcopy(trainY0))

    for i in range(PERIODS-1):
        trainX.append(copy.deepcopy(trainY[i]))
        trainX_all.append(copy.deepcopy(trainX))
        trainY0.pop(0)
        trainY_all.append(copy.deepcopy(trainY0))
# trainX_all, trainY_all    为用于滚动分解的X和Y序列


    return trainX_all, trainY_all

def Roll_create_dateback(df, DATE_BACK ,ahead=1):   #按训练集的规则进行标准化，后续测试集加入的数据都按照此规则进行标准化

    # Normalize without unifying

    trainY = np.array(df.values).reshape(-1, 1)
    trainX = trainY

    # Create dateback
    dataX, dataY = [], []
    ahead = ahead - 1
    for i in range(len(trainY) - DATE_BACK - ahead):
        dataX.append(copy.deepcopy(np.array(trainX[i:(i + DATE_BACK)])))
        dataY.append(copy.deepcopy(np.array(trainY[i + DATE_BACK + ahead])))
    return np.array(dataX), np.array(dataY), np.array(trainX[-DATE_BACK:])

def Roll_create_dateback_multivar(data, DATE_BACK, ahead=1, order=0): 
    "生成样本对，例如，用所有变量的前10个来预测目标变量的第11个值"
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



# Rolldecom LSTM Forecasting with 3 Co-IMFs
# -------------------------------
# It uses LSTM directly for prediction wiht input_shape=[DATE_BACK,the number of features]
def Rolldecom_Respect_ANN(data=None, Roll_type='type1', MODEL_type=None, draw=True, uni=False, show_model=True, next_pred=True, ahead=1):
    print('==============================================================================================')
    print('This is Rolldecom LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input dataset and load
    input_df, file_name = check_dataset(data, input_form='series', uni_nor=uni)  # include check_vars()

    #按滚动分析模式产生待分解数据及其对应测试集
    testX_all, testY_all = Rolldecom(data, PERIODS)  #这里输入的data应该是Series格式数据

    # Create ans show the inputting data set
    #开始逐步分解，并在每一步分解建立Respect ANN
    LSTM_MODEL = MODEL_type
    declare_LSTM_MODEL(model=LSTM_MODEL)  #通过该函数声明这里调用的函数类型
    if Roll_type == 'type1':  #type1指正常滚动分解，适用于每次分解，每次训练的简单模型
        test_pred = []
        start = time.time()
        for i in range(len(testX_all)):
            temp_data  =  copy.deepcopy(pd.Series(testX_all[i]))
            temp_data.name = copy.deepcopy('第%s次滚动分解'%i)
            start_Internal = time.time()
            print(temp_data.name)
            decom_data = emd_decom(series= temp_data, draw=False  )
            trainX, trainY, scalarY, next_trainX = Roll_create_dateback(decom_data, ahead=ahead)
            #在当前步骤进行一步预测

            print('\nInput Shape: (%d,%d)\n' % (trainX.shape[1], trainX.shape[2]))

            model = LSTM_model(trainX.shape)
            if show_model: model.summary()  # The summary of layers and parameters

            history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_split=VALIDATION_SPLIT,
                                verbose=VERBOSE, shuffle=SHUFFLE)

            if next_pred:  # predict tomorrow not in test set
                next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
                next_ans_out = scalarY.inverse_transform(copy.deepcopy(next_ans))
                print('当前预测结果为：',next_ans_out)
            end_Internal = time.time()
            print('该轮分解的Running time: %.3fs' % (end_Internal - start_Internal))
        test_pred.append(copy.deepcopy(next_ans_out))
        df_pred = pd.DataFrame(test_pred)

        end = time.time()
        df_pred = pd.DataFrame(test_pred)
        pd.DataFrame.to_csv(df_pred, LOG_PATH + file_name + 'ensemble_' + MODE + FORM + '_pred.csv')


    if Roll_type == 'type2':  #type2指正常滚动分解，后续预测都直接应用预训练好的模型进行预测（如果IMFs数量与训练时不等，利用熵来合并相近序列）
        test_pred = []
        start = time.time()
        for i in range(len(testX_all)):
            temp_data  =  copy.deepcopy(pd.Series(testX_all[i]))
            temp_data.name = copy.deepcopy('第%s次滚动分解'%i)
            start_Internal = time.time()
            print(temp_data.name)
            decom_data = emd_decom(series= temp_data, draw=False  )
            trainX, trainY, scalarY, next_trainX = Roll_create_dateback(decom_data, ahead=ahead)
            #在当前步骤进行一步预测

            print('\nInput Shape: (%d,%d)\n' % (trainX.shape[1], trainX.shape[2]))

            model = LSTM_model(trainX.shape)
            if show_model: model.summary()  # The summary of layers and parameters
            if i == 0:
                history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                    validation_split=VALIDATION_SPLIT,
                                    verbose=VERBOSE, shuffle=SHUFFLE)

                if next_pred:  # predict tomorrow not in test set
                    next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
                    next_ans_out = scalarY.inverse_transform(copy.deepcopy(next_ans))
                    print('当前预测结果为：', next_ans_out)

            else:
                next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
                next_ans_out = scalarY.inverse_transform(copy.deepcopy(next_ans))
                print('当前预测结果为：', next_ans_out)

            end_Internal = time.time()
            print('该轮分解的Running time: %.3fs' % (end_Internal - start_Internal))
        test_pred.append(copy.deepcopy(next_ans_out))
        df_pred = pd.DataFrame(test_pred)

        end = time.time()
        df_pred = pd.DataFrame(test_pred)
        pd.DataFrame.to_csv(df_pred, LOG_PATH + file_name + 'ensemble_' + MODE + FORM + '_pred.csv')
    # Evaluate model
    if PERIODS != 0:
        if draw and file_name == '': plot_all('Rolldecom_Respect_ANN', test_pred)  # plot chart to campare
        df_evl = evl(input_df['sum'][-PERIODS:].values, test_pred, scale='input df')
        print('Running time: %.3fs' % (end - start))
        df_evl.append(end - start)
        df_evl = pd.DataFrame(df_evl).T  # ['R2','RMSE','MAE','MAPE','Time']
        if next_pred:
            print('##################################')
            print('Today is', input_df['sum'][-1:].values, 'but predict as', df_pred[-2:-1].values)
            print('Next day is', df_pred[-1:].values)
        pd.DataFrame.to_csv(df_evl, LOG_PATH + file_name + 'ensemble_' + MODE + FORM + '_log.csv', index=False,
                            header=0, mode='a')  # log record
        print('Ensemble LSTM Forecasting finished, check the logs',
              LOG_PATH + file_name + 'ensemble_' + MODE + FORM + '_log.csv')
    return df_pred

#========================================滚动分解================================================================


# Multiple predictions
# -------------------------------
# Each Multi_pred() takes long time to run around 1000s unless setting the EPOCHS and n.
class HiddenPrints:  # used to hide the print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def Multi_pred(df=None, run_times=10, uni_nor=False, single_lstm=False, ensemble_lstm=False, respective_lstm=False,
               hybrid_lstm=False, redecom=None, ahead=1):
    print('Multiple predictions of ' + str.upper(MODE) + FORM + ' is running...')
    input_df, file_name = check_dataset(df, input_form='df', use_series=True, uni_nor=uni_nor)  # include check_vars()
    if file_name == '':
        input_series = None
    else:
        input_series = input_df.T.sum()
    start = time.time()
    with HiddenPrints():
        for i in range(run_times):
            if single_lstm: Single_LSTM(series=input_series, draw=False, uni=uni_nor, ahead=ahead)
            if ensemble_lstm: Ensemble_LSTM(df=df, draw=False, uni=uni_nor, ahead=ahead)
            if respective_lstm: Respective_LSTM(df=df, draw=False, uni=uni_nor, ahead=ahead)
            if hybrid_lstm: Hybrid_LSTM(df=df, draw=False, redecom=redecom, ahead=ahead)
    end = time.time()
    print('Multiple predictions completed, taking %.3fs' % (end - start))
    print('Please check the logs in: ' + LOG_PATH)


# 6.Hybrid Forecasting Functions
# Please use cl.declare_vars() to determine variables.
# ==============================================================================================


# 7.Statistical Tests
# Please use cl.declare_vars() to determine variables.
# ==============================================================================================
def statistical_tests(series=None):  # total version
    input_series, file_name = check_dataset(series, input_form='series', use_series=True)  # include check_vars()
    adf_test(input_series)
    print()
    LB_test(input_series)
    print()
    JB_test(input_series)
    print()
    plot_acf_pacf(input_series)


# Augmented Dickey-Fuller test (ADF test) for stationarity
# -------------------------------
def adf_test(series=None):
    if series is None: raise ValueError('This is no proper input.')
    adf_ans = adfuller(series)  # The outcomes are test value, p-value, lags, degree of freedom.
    print('##################################')
    print('ADF Test')
    print('##################################')
    print('Test value:', adf_ans[0])
    print('P value:', adf_ans[1])
    print('Lags:', adf_ans[2])
    print('1% confidence interval:', adf_ans[4]['1%'])
    print('5% confidence interval:', adf_ans[4]['5%'])
    print('10% confidence interval:', adf_ans[4]['10%'])
    # print(adf_ans)

    # Brief review
    adf_status = ''
    if adf_ans[0] <= adf_ans[4]['1%']:
        adf_status = 'very strong'
    elif adf_ans[0] <= adf_ans[4]['5%']:
        adf_status = 'strong'
    elif adf_ans[0] <= adf_ans[4]['10%']:
        adf_status = 'normal'
    else:
        adf_status = 'no'
    print('The p-value is ' + str(adf_ans[1]) + ', so the series has ' + str(adf_status) + ' stationarity.')
    print('The automatic selecting lags is ' + str(adf_ans[2]) + ', advising the past ' + str(
        adf_ans[2]) + ' days as the features.')


# Ljung-Box Test for autocorrelation
# -------------------------------
def LB_test(series=None):
    if series is None: raise ValueError('This is no proper input.')
    lb_ans = lb_test(series, lags=None, boxpierce=False)  # The default lags=40 for long series.
    print('##################################')
    print('Ljung-Box Test')
    print('##################################')

    # Plot p-values in a figure
    fig = plt.figure(figsize=(10, 3))
    pd.Series(lb_ans[1]).plot(label="Ljung-Box Test p-values")
    plt.xlabel('Lag')
    plt.legend()
    plt.show()

    # Brief review
    if np.sum(lb_ans[1]) <= 0.05:
        print('The sum of p-value is ' + str(np.sum(
            lb_ans[1])) + '<=0.05, rejecting the null hypothesis that the series has very strong autocorrelation.')
    else:
        print('Please view with the line chart, the autocorrelation of the series may be not strong.')

    # Show the outcome
    # print(pd.DataFrame(lb_ans)) # The outcomes are test value at line 0, and p-value at line 1.


# Jarque-Bera Test for normality
# -------------------------------
def JB_test(series=None):
    if series is None: raise ValueError('This is no proper input.')
    jb_ans = jb_test(series)  # The outcomes are test value, p-value, skewness and kurtosis.
    print('##################################')
    print('Jarque-Bera Test')
    print('##################################')
    print('Test value:', jb_ans[0])
    print('P value:', jb_ans[1])
    print('Skewness:', jb_ans[2])
    print('Kurtosis:', jb_ans[3])

    # Brief review
    if jb_ans[1] <= 0.05:
        print(
            'p-value is ' + str(jb_ans[1]) + '<=0.05, rejecting the null hypothesis that the series has no normality.')
    else:
        print('p-value is ' + str(
            jb_ans[1]) + '>=0.05, accepting the null hypothesis that the series has certain normality.')


# Plot ACF and PACF figures
# -------------------------------
def plot_acf_pacf(series=None):
    print('##################################')
    print('ACF and PACF')
    print('##################################')
    if series is None: raise ValueError('This is no proper input.')
    fig = plt.figure(figsize=(10, 5))
    fig1 = fig.add_subplot(211)
    plot_acf(series, lags=40, ax=fig1)
    fig2 = fig.add_subplot(212)
    plot_pacf(series, lags=40, ax=fig2)

    # Save the figure
    plt.savefig(FIGURE_PATH + 'ACF and PACF of Series.svg', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# Single ARIMA 模型
# -------------------------------
from statsmodels.tsa.stattools import adfuller  # adf_test
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test  # LB_test
from statsmodels.stats.stattools import jarque_bera as jb_test  # JB_test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # plot_acf_pacf
import statsmodels.api as sm

#from statsmodels.tsa import arma_order_select_ic
def Single_ARIMA(series=None, draw = False):
    start = time.time()
    predict_test = []
    plt.plot(series)
    print('==================将原数据按PERIODS:%s划分为训练集和测试集====================='%PERIODS)
    train = series[:-PERIODS]
    test  = series[-PERIODS:]
    F = adfuller(train)
    print('对训练集进行单位根检验,分别输出1%、%5、%10不同程度拒绝原假设的统计值',F)
    # 白噪声检验
    print('对训练集进行白噪声检验,分别输出6、12阶下的LB和BP统计量,若P值小于0.05则拒绝原假设,认为原序列为非白噪声序列',lb_test(train, lags = [6, 12],boxpierce=True))
    
    if F[0] >= F[4]['5%']:  #单位根检验值大于5%置信条件下的P值
        for i in range(5):
            d = i+1
            train = np.diff(train)
            temp_F = adfuller(train)
            if temp_F[0] < temp_F[4]['5%']:
                break
            else:
                continue
    else:
        d = 0
    #绘制自相关和偏自相关图
    if draw:
        acf=plot_acf(train)
        pacf=plot_pacf(train)

    #通过BIC准则来选择最合适的AR和MA阶数
    trend_evaluate =  sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=5, max_ma=5)

    (p,q) = trend_evaluate.bic_min_order   #aic_min_order
    print('train AIC', trend_evaluate.aic_min_order)
    print('train BIC', trend_evaluate.bic_min_order)
    for i in range(PERIODS):
        train = series[:-PERIODS+i]
        model = sm.tsa.arima.ARIMA(train,order=(p,d,q))
        arima_res = model.fit()
        arima_res.summary()
        predict=arima_res.predict(train.index[-1])

        predict_test.append(copy.deepcopy(predict))
    end = time.time()
    print(len(predict_test))
    
    evl(test, predict_test)
    print('Running time: %.3fs' % (end - start))






# 8.Forecasting of SVR
# Please use cl.declare_vars() to determine variables.
# ==============================================================================================
 
#from sklearn.svm import SVR
#from thundersvm import *
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV


def Normalize(series_train= None, series_test= None):
    max = np.max(series_train)
    min = np.min(series_train)
    trainx = [(x-min)/(max-min) for x in series_train]
    trainy = [(y-min)/(max-min) for y in series_test]

    return  trainx, trainy, (max, min)

def Roll_Respective_SVR(series=None):  #建立一个对原序列进行滚动分解，且每一步进行调参的Respective_SVR模型
    test_pred = []
    test_pred_all = np.zeros([100,20])
    testX_all, testY_all = Rolldecom(series, PERIODS) 
    for i in range(len(testX_all)):

        temp_data = pd.Series(testX_all[i])
        decom_data = emd_decom(series=temp_data, draw=False)
        #得到该次滚动分解的IMFs序列，现对各序列分别建立SVR模型
        Onestep_predict = 0  #各IMFs加总得到每一步预测值                                              
        
        print('该次分解所得数据的shape为：',(decom_data.shape[0],decom_data.shape[1]))
        for j in range(decom_data.shape[1]):
            #存放每一个IMF的预测结果

            print('========第%s轮的第%s个IMF序列开始建模============='%(i+1,j+1))
            single_decom_data = decom_data.iloc[:,j]
            #得到该次分解的第j个IMF的预测值
            ans_pred = SVR_pred(data=single_decom_data, gstimes=3, draw=False, ahead=1, Roll_type = 'None',next_pred = True)

            test_pred_all[i,j] = ans_pred.ravel()
            Onestep_predict = copy.deepcopy(Onestep_predict) + ans_pred.ravel() 
        print('======================该轮分解的预测值为========================:',Onestep_predict)
        #得到该次分解的最终数据（由各个IMF预测结果加法加总得到）
        test_pred.append(Onestep_predict)
        print('目前已得到预测值为：',test_pred)
    return test_pred, test_pred_all


def SVR_pred(data=None, gstimes=5, draw=True, ahead=1, Roll_type = 'type1',next_pred = False):
    import pylab
    # 按滚动分析模式产生待分解数据及其对应测试集
    testX_all, testY_all = Rolldecom(data, PERIODS)  # 这里输入的data应该是Series格式数据
    # Grid Search of K-Fold CV
    # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts


  # TYPE == None 指不进行序列分解，直接使用Single_SVR对原序列进行预测，并通过网格搜索参数进行测试集(长度为PERIODS)或进行一步next_pred预测
    if Roll_type == 'None':
        if next_pred == False:

            #利用训练集特征对训练集和测试集进行标准化
            dataX, dataY, (max, min) = Normalize(series_train= data[:-PERIODS], series_test= data[-PERIODS:])
            dataX.extend(dataY)
            dataX = pd.Series(dataX)
            start = time.time()
            #生成训练、测试样本对
            trainX,trainY,next_trainX = Roll_create_dateback(dataX,ahead=ahead)

            trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]))
            x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
            y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]

            #Grid Search of K-Fold CV
            #logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
            C_range = np.logspace(-2, 10, 26)
            gamma_range = np.logspace(-9, 3, 26)
            best_gamma,best_C = 0,0
            for i in range(gstimes):
                print('Iteration',i)
                param_grid = dict(gamma=gamma_range, C=C_range)
                grid = GridSearchCV(SVR(), param_grid=param_grid, cv=3)
                grid.fit(x_train, y_train)
                
                print('Best parameters:', grid.best_params_)
                if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']: break
                best_gamma=grid.best_params_['gamma']
                best_C=grid.best_params_['C']
                gamma_range = np.append(np.linspace(best_gamma/10,best_gamma*0.9,9),np.linspace(best_gamma,best_gamma*10,10)).ravel()
                C_range = np.append(np.linspace(best_C/10,best_C*0.9,9),np.linspace(best_C,best_C*10,10)).ravel()

                #best_gamma,best_C = 3.792887875145918e-07, 478630.092322638
                # Predict
            clf = SVR(kernel='rbf', gamma=best_gamma ,C=best_C)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            print('y_pred is :',y_pred)
            end = time.time()
            evl(dataY, test_pred)

            # De-normalize and Evaluate
            test_pred = list( y_pred.ravel() )
            test_pred = [(x*(max-min)+min ) for x in test_pred]

            test_y = list(y_test)
            test_y = [(x*(max-min)+min ) for x in test_y]

            print('Running time: %.3fs'%(end-start))
        else:
            #该状态是进行next_pred ，即输入的数据data全是训练集，对data后一步进行预测，只输出预测值，需在外界函数对预测结果进行评估
            start = time.time()
            trainX,trainY,scalarY,next_trainX = create_dateback(data,ahead=ahead)
            trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]))
            x_train = trainX
            y_train = trainY

            #Grid Search of K-Fold CV
            #logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
            C_range = np.logspace(-2, 10, 26)
            gamma_range = np.logspace(-9, 3, 26)
            best_gamma,best_C = 0,0
            for i in range(gstimes):
                print('Iteration',i)
                param_grid = dict(gamma=gamma_range, C=C_range)
                grid = GridSearchCV(SVR(), param_grid=param_grid, cv=3)
                grid.fit(x_train, y_train)

                print('Best parameters:', grid.best_params_)
                if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']: break
                best_gamma=grid.best_params_['gamma']
                best_C=grid.best_params_['C']
                gamma_range = np.append(np.linspace(best_gamma/10,best_gamma*0.9,9),np.linspace(best_gamma,best_gamma*10,10)).ravel()
                C_range = np.append(np.linspace(best_C/10,best_C*0.9,9),np.linspace(best_C,best_C*10,10)).ravel()

                #best_gamma,best_C = 3.792887875145918e-07, 478630.092322638
                # Predict
            clf = SVR(kernel='rbf', gamma=best_gamma ,C=best_C)
            clf.fit(x_train, y_train)
            next_trainX = np.array(next_trainX).reshape(1,-1)
            y_pred = clf.predict(next_trainX)
            print('y_pred is :',y_pred)
            end = time.time()

            # De-normalize and Evaluate
            test_pred = scalarY.inverse_transform(y_pred.reshape(y_pred.shape[0],1))

        print('预测结果为：', test_pred)
        print('Running time: %.3fs'%(end-start))











    if Roll_type == 'type1':  # type1指正常滚动分解,但仅在第一次分解时或IMFs数量不一致时进行参数搜索的预测方法，参数采取网格搜索

        start = time.time()
        test_pred = []
        #滚动分解
        
        all_params_extend = []
        for i in range(len(testX_all)):
            temp_data  =  copy.deepcopy(pd.Series(testX_all[i]))
            temp_data.name = copy.deepcopy('已准备好第%s次进行滚动分解的训练数据集'%(i+1))
            start_Internal = time.time()
            print(temp_data.name)
            decom_data = emd_decom(series=temp_data, draw=False)
            #得到该次滚动分解的IMFs序列，现对各序列分别建立SVR模型
            Onestep_predict = 0  #各IMFs加总得到每一步预测值                                              
            all_params = []
            print('该次分解所得数据的shape为：',(decom_data.shape[0],decom_data.shape[1]))
            for j in range(decom_data.shape[1]):
                #存放每一个IMF的预测结果

                print('========第%s轮的第%s个IMF序列开始建模============='%(i+1,j+1))
                single_decom_data = decom_data.iloc[:,j]
                #对IMF序列进行标准化，构建训练和测试对（注意：是每一个IMF单独标准化，那么还原时需对应）
                trainX, trainY, scalarY, next_trainX = create_dateback(single_decom_data, ahead=ahead)
                trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]))
                C_range = np.logspace(-2, 10, 26)
                gamma_range = np.logspace(-9, 3, 26)
                
                if i == 0:
                    #Grid search the best params on the first time 
                    best_gamma, best_C = 0, 0
                    print('best params:%s:' % [best_gamma, best_C])
                    for k in range(gstimes):
                        param_grid = dict(gamma=gamma_range, C=C_range)
                        print('Iteration', k)
                        grid = GridSearchCV(SVR(), param_grid=param_grid, cv=5)
                        grid.fit(trainX, trainY)
                        print('Best parameters:', grid.best_params_)
                        if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']:
                             break
                        best_gamma = grid.best_params_['gamma']
                        best_C = grid.best_params_['C']
                        gamma_range = np.append(np.linspace(best_gamma / 10, best_gamma * 0.9, 9),
                                                np.linspace(best_gamma, best_gamma * 10, 10)).ravel()
                        C_range = np.append(np.linspace(best_C / 10, best_C * 0.9, 9), np.linspace(best_C, best_C * 10, 10)).ravel()
                    
                    #Saving the best params of each SVM model. Notice: the jth parameter corresponds to the model of this jth IMF
                    all_params.append([copy.deepcopy(best_gamma),copy.deepcopy(best_C)])

                    # Predict
                    print('开始拟合SVR模型')
                    clf = SVR(kernel='rbf', gamma=best_gamma, C=best_C)
                    clf.fit(trainX, trainY)
                    next_trainX = next_trainX.reshape(-1, trainX.shape[1])
                    print('开始预测')
                    y_pred = clf.predict(next_trainX)
                    # saving the params of models
                    if len(all_params) == decom_data.shape[1]:
                        print('该数量的IMFs参数组合已经训练完，储存至all_params_extend')
                        all_params_extend.append(copy.deepcopy(all_params))
            
                else:
                    #Predict
               
                    for item in all_params_extend:
                        #判断是否已储存该IMFs数量的模型参数
                        if decom_data.shape[1] == len(item):
                            print('There are already combinations of parameters in the portfolio that fit the following number of IMFs',len(item))
                            all_params = item

                    if len(all_params) != decom_data.shape[1]:


                        print('================The nums of IMFs is%s, not consistent with len of the exists all_params========================='%decom_data.shape[1])
                        print('Show exits all_params_extend:',all_params_extend)
                        print('Start a new training session')

                        for k in range(gstimes):
                            param_grid = dict(gamma=gamma_range, C=C_range)
                            grid = GridSearchCV(SVR(), param_grid=param_grid, cv=10)
                            grid.fit(trainX, trainY)
                            print('Iteration', k)
                            print('Best parameters:', grid.best_params_)
                            if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']:
                                break
                            best_gamma = grid.best_params_['gamma']
                            best_C = grid.best_params_['C']
                            gamma_range = np.append(np.linspace(best_gamma / 10, best_gamma * 0.9, 9),
                                                    np.linspace(best_gamma, best_gamma * 10, 10)).ravel()
                            C_range = np.append(np.linspace(best_C / 10, best_C * 0.9, 9), np.linspace(best_C, best_C * 10, 10)).ravel()
                        #Saving the best params of each SVM model. Notice: the jth parameter corresponds to the model of this jth IMF
                        all_params.append([copy.deepcopy(best_gamma),copy.deepcopy(best_C)])

                         # Predict
                        print('开始拟合SVR模型')
                        clf = SVR(kernel='rbf', gamma=best_gamma, C=best_C)
                        clf.fit(trainX, trainY)
                        next_trainX = next_trainX.reshape(-1, trainX.shape[1])
                        print('开始预测')
                        y_pred = clf.predict(next_trainX)
                        # saving the params of models
                        if len(all_params) == decom_data.shape[1]:
                            print('该数量的IMFs参数组合已经训练完，储存至all_params_extend')
                            all_params_extend.append(copy.deepcopy(all_params))

                    else:
                        #Using the best params for this model 
                        [best_gamma, best_C] = all_params[j]
                        #Model traing with newdata,but without params grid search 
                        clf = SVR(kernel='rbf', gamma=best_gamma, C=best_C)
                        clf.fit(trainX, trainY)
                        next_trainX = next_trainX.reshape(-1, trainX.shape[1])
                        y_pred = clf.predict(next_trainX)

                #对Single_IMF这次滚动分解的结果进行反标准化 De-normalize
                #reshape the y_pred to (-1,1) or it will not be de-normalized
                y_pred = np.array(y_pred).reshape(-1,1)
                y_pred = scalarY.inverse_transform(copy.deepcopy(y_pred))
                Onestep_predict = copy.deepcopy(Onestep_predict) + y_pred[0][0]
                end_Internal = time.time()
                print('This Step Running time: %.3fs' % (end_Internal - start_Internal))


            test_pred.append(copy.deepcopy(Onestep_predict))
        # Evaluate
        df_pred = pd.DataFrame(test_pred)
        pd.DataFrame.to_csv(df_pred, LOG_PATH + SERIES.name + '_SVR_pred.csv')
        test_y = data[-PERIODS:]
        evl(test_y,test_pred)
        end = time.time()
        print('Running time: %.3fs' % (end - start))

  
        



    if Roll_type == 'type2':  # type2指正常滚动分解,同时每一轮都重新训练每个IMFs的SVM参数

        for i in range(len(testX_all)):
            temp_data  =  copy.deepcopy(pd.Series(testX_all[i]))
            temp_data.name = copy.deepcopy('已准备好第%s次进行滚动分解的训练数据集'%(i+1))
            start_Internal = time.time()
            print(temp_data.name)
            decom_data = emd_decom(series=temp_data, draw=False)
            #得到该次滚动分解的IMFs序列，现对各序列分别建立SVR模型
            Onestep_predict = 0  #各IMFs加总得到每一步预测值                                              

            print('该次分解所得数据的shape为：',(decom_data.shape[0],decom_data.shape[1]))
            test_pred = []
            start = time.time()
        
        #每个SVR模型的参数设定为如下固定值

            #滚动分解
            start = time.time()
            trainX,trainY,scalarY,next_trainX = create_dateback(data,ahead=ahead)
            trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]))
            x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
            y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]

            #Grid Search of K-Fold CV
            #logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
            C_range = np.logspace(-2, 10, 26)
            gamma_range = np.logspace(-9, 3, 26)
            best_gamma,best_C = 0,0
            for i in range(gstimes):
                param_grid = dict(gamma=gamma_range, C=C_range)
                grid = GridSearchCV(SVR(), param_grid=param_grid, cv=10)
                grid.fit(x_train, y_train)
                print('Iteration',i)
                print('Best parameters:', grid.best_params_)
                if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']: break
                best_gamma=grid.best_params_['gamma']
                best_C=grid.best_params_['C']
                gamma_range = np.append(np.linspace(best_gamma/10,best_gamma*0.9,9),np.linspace(best_gamma,best_gamma*10,10)).ravel()
                C_range = np.append(np.linspace(best_C/10,best_C*0.9,9),np.linspace(best_C,best_C*10,10)).ravel()

            #best_gamma,best_C = 3.792887875145918e-07, 478630.092322638
            # Predict
            clf = SVR(kernel='rbf', gamma=best_gamma ,C=best_C)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            end = time.time()

        # De-normalize and Evaluate
        test_pred = scalarY.inverse_transform(y_pred.reshape(y_pred.shape[0],1))
        test_y = scalarY.inverse_transform(y_test)
        evl(test_y, test_pred)
        print('Running time: %.3fs'%(end-start))

    # Plot observation figures
    if draw:
        fig = plt.figure(figsize=(5,2))
        plt.plot(test_y)
        plt.plot(test_pred)
        plt.title('SVR forecasting result')
        #plt.savefig(FIGURE_PATH+fig_name+' LSTM forecasting result.svg', bbox_inches='tight')
        plt.show()


    return test_pred




# 9.Forecasting of hybrid ensemble model
# Please use cl.declare_vars() to determine variables.
# ==============================================================================================


def Roll_Respective_SVR(series=None):  #建立一个对原序列进行滚动分解，且每一步进行调参的Respective_SVR模型
    test_pred = []
    test_pred_all = np.zeros([100,20])
    testX_all, testY_all = Rolldecom(series, PERIODS) 
    for i in range(len(testX_all)):

        temp_data = pd.Series(testX_all[i])
        decom_data = emd_decom(series=temp_data, draw=False)
        #得到该次滚动分解的IMFs序列，现对各序列分别建立SVR模型
        Onestep_predict = 0  #各IMFs加总得到每一步预测值                                              
        
        print('该次分解所得数据的shape为：',(decom_data.shape[0],decom_data.shape[1]))
        for j in range(decom_data.shape[1]):
            #存放每一个IMF的预测结果

            print('========第%s轮的第%s个IMF序列开始建模============='%(i+1,j+1))
            single_decom_data = decom_data.iloc[:,j]
            #得到该次分解的第j个IMF的预测值
            ans_pred = SVR_pred(data=single_decom_data, gstimes=3, draw=False, ahead=1, Roll_type = 'None',next_pred = True)

            test_pred_all[i,j] = ans_pred.ravel()
            Onestep_predict = copy.deepcopy(Onestep_predict) + ans_pred.ravel() 
        print('======================该轮分解的预测值为========================:',Onestep_predict)
        #得到该次分解的最终数据（由各个IMF预测结果加法加总得到）
        test_pred.append(Onestep_predict)
        print('目前已得到预测值为：',test_pred)
    return test_pred, test_pred_all

# ========================================================================================================================
#建立一个对原序列进行滚动分解，且通过样本熵重构序列，并利用多模型进行集成预测的混合模型框架
def Roll_Models_Ensemble(series=None): 
    test_pred = []
    test_pred_all = np.zeros([100,20])
    testX_all, testY_all = Rolldecom(series, PERIODS) 
    for i in range(len(testX_all)):

        temp_data = pd.Series(testX_all[i])
        decom_data = emd_decom(series=temp_data, draw=False)
        #得到该次滚动分解的IMFs序列，现对各序列分别建立SVR模型
        Onestep_predict = 0  #各IMFs加总得到每一步预测值                                              
        
        print('该次分解所得数据的shape为：',(decom_data.shape[0],decom_data.shape[1]))
        #对分解数据进行自动样本熵重构
        group_all, ave_se = Auto_Se_Group(imfs_df=decom_data)

        for j in range(decom_data.shape[1]):
            #存放每一个IMF的预测结果

            print('========第%s轮的第%s个IMF序列开始建模============='%(i+1,j+1))
            single_decom_data = decom_data.iloc[:,j]
            #得到该次分解的第j个IMF的预测值
            ans_pred = SVR_pred(data=single_decom_data, gstimes=3, draw=False, ahead=1, Roll_type = 'None',next_pred = True)

            test_pred_all[i,j] = ans_pred.ravel()
            Onestep_predict = copy.deepcopy(Onestep_predict) + ans_pred.ravel() 
        print('======================该轮分解的预测值为========================:',Onestep_predict)
        #得到该次分解的最终数据（由各个IMF预测结果加法加总得到）
        test_pred.append(Onestep_predict)
        print('目前已得到预测值为：',test_pred)
    return test_pred, test_pred_all






# VMD # There are some problems in this module
# -------------------------------
def vmd_decom(series=None,alpha=2000,tau=0,K=5,DC=0,init=1,tol=1e-7,re_decom=True,re_imf=0,draw=True):
    # Check input
    dataset,file_name = check_dataset(series,input_form='series') # include check_vars()

    from vmdpy import VMD  
    # VMD parameters
    #alpha = 2000       # moderate bandwidth constraint  
    #tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
    #K = 3              # 3 modes  
    #DC = 0             # no DC part imposed  
    #init = 1           # initialize omegas uniformly  
    #tol = 1e-7         

    # VMD 
    imfs_vmd, imfs_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)  
    imfs_num = np.shape(imfs_vmd)[0]
    
    if draw:
        # Plot original data
        series_index = range(len(series))
        fig = plt.figure(figsize=(16,2*imfs_num))
        plt.subplot(1+imfs_num, 1, 1 )
        plt.plot(series_index, series, color='#0070C0') #F27F19 orange #0070C0 blue
        plt.ylabel('VMD Original data')
    
        # Plot IMFs
        for i in range(imfs_num):
            plt.subplot(1 + imfs_num,1,2 + i)
            plt.plot(series_index, imfs_vmd[i, :], color='#F27F19')
            plt.ylabel('VMD-IMF'+str(i))

        # Save figure
        fig.align_labels()
        plt.tight_layout()
        if (re_decom==False): plt.savefig(FIGURE_PATH+file_name+'VMD Result.svg', bbox_inches='tight')
        else: plt.savefig(FIGURE_PATH+'IMF'+str(re_imf)+' VMD Re-decomposition Result.svg', bbox_inches='tight')
        plt.show()
    

        #绘制频谱图
        N= len(series)
        Fs =   1            #12000
        Ts = 1.0/Fs
        t = np.arange(N)
        small_k = np.arange(N)
        T = N/Fs
        frq = small_k/T
        frq1 = frq[range(int(N/2))]
        
        
        plt.figure(figsize=(10, 8))
        for i in range(K):
            
            plt.subplot(K + 1, 1, 1)
            data_f = abs(np.fft.fft(series)) / N
            data_f1 = data_f[range(int(N / 2))]
            plt.plot(frq1, data_f1)
            plt.title("outer")
            plt.subplot(K+1, 1, i + 2)
            data_f2 = abs(np.fft.fft(imfs_vmd[i, :])) / N
            data_f3 = data_f2[range(int(N / 2))]
            plt.plot(frq1, data_f3, 'red')
            plt.xlabel('pinlv(hz)')
            plt.ylabel('u{}'.format(i + 1))
        plt.tight_layout()
        plt.show()




    # Save data
    imfs_df = pd.DataFrame(imfs_vmd.T)
    imfs_df.columns = ['imf'+str(i) for i in range(imfs_num)]
    if file_name == '':
        if (re_decom==False): 
            pd.DataFrame.to_csv(imfs_df,PATH+file_name+'vmd_data.csv')
            print('VMD finished, check the dataset: ',PATH+file_name+'vmd_data.csv')

    return imfs_df # pd.DataFrame















# DM test # Author: John Tsang
def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt, msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt, msg)
        len_act = len(actual_lst)
        len_p1 = len(pred1_lst)
        len_p2 = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt, msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt, msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt, msg)
            # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")

        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True

        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt, msg)
        return (rt, msg)

    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections

    # Initialise lists
    e1_lst, e2_lst, d_lst = [], [], []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    T = float(len(actual_lst))

    # construct d according to crit
    if (crit == "MSE"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append((actual - p1) ** 2)
            e2_lst.append((actual - p2) ** 2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs((actual - p1) / actual))
            e2_lst.append(abs((actual - p2) / actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(((actual - p1)) ** (power))
            e2_lst.append(((actual - p2)) ** (power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

            # Mean of d
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / (T)) * autoCov

    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    DM_stat = V_d ** (-0.5) * mean_d
    harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
    DM_stat = harvey_adj * DM_stat
    # Find p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM=DM_stat, p_value=p_value)
    return rt


