# MA-EMD: Aligned Empirical Decomposition for Multivariate Time-Series Forecasting
## Citation Statement
If this code is used for research, academic publication, or any form of application, please be sure to cite the following paper:Xiangjun Cai, Dagang Li, Jinglin Zhang, Zhuohao Wu. MA-EMD: Aligned empirical decomposition for multivariate time-series forecasting. Expert Systems With Applications, 267(2025): 126080.DOI: https://doi.org/10.1016/j.eswa.2024.126080
## Code Functionality
This code implements the MA-EMD (Multivariate Aligned Empirical Mode Decomposition) method proposed in the paper, which is designed to optimize the performance of Decomposition-Ensemble Models (DEMs) in multivariate time-series forecasting.

The core functionalities include:
1. Performing standard EMD decomposition on each variable of the multivariate time series independently to ensure decomposition quality;
2. Achieving frequency alignment of Intrinsic Mode Functions (IMFs) (decomposed subsequences) across different variables using Kullback-Leibler (KL) divergence, based on the intrinsic frequency components of the target variable;
3. Constructing an MA-EMD-based decomposition-ensemble forecasting model to improve both prediction accuracy and computational efficiency;
4. Supporting integration with mainstream predictors such as SVR and LSTM, and being applicable to multivariate time-series forecasting tasks in various fields including climate, power systems, and exchange rates.

## Core Theories of the Paper
1. Limitations of Existing Methods

To achieve alignment of decomposed subsequences across multiple variables, the traditional Multivariate Empirical Mode Decomposition (MEMD) needs to relax decomposition quality constraints. This leads to the generation of low-quality and unnecessary frequency components, accompanied by a significant increase in computational overhead, ultimately impairing forecasting performance.

2. Core Innovations of MA-EMD

Independent High-Quality Decomposition: Each variable is decomposed independently using standard EMD, strictly ensuring the periodicity and envelope symmetry of IMFs and avoiding the quality loss associated with MEMD;

Target-Oriented Alignment Mechanism: Using the IMFs of the target variable as a reference, the frequency characteristics of IMFs are represented by the distribution of extremum intervals. KL divergence is employed to measure and match the frequency components of IMFs from other variables, enabling accurate alignment;

Partial Decomposition Optimization: A threshold ω is introduced to control the validity of IMFs, filtering out low-quality IMFs with too few extrema to balance the completeness of decomposition and the stability of forecasting.

3. Model Framework

Data Standardization: Convert the multivariate time series into a format with zero mean and unit variance;

EMD Decomposition: Perform EMD on each variable independently to obtain a set of IMFs for each variable;

Frequency Alignment: Group the IMFs of related variables with those of the target variable based on KL divergence;

Sub-Model Training: Train an independent multivariate forecasting sub-model for each aligned IMF group;

Result Ensemble: Aggregate the prediction results of all sub-models to obtain the final forecast for the target variable.

## Installation Environment Requirements
### Basic Environment

Python Version: 3.7.3 (installation via Anaconda is recommended);

Development Tool: PyCharm 2019.2.3 (Community Edition, optional);

Hardware Support: An NVIDIA GPU (e.g., RTX 3090 Ti) is recommended to accelerate model training. The code can run on a CPU but with lower efficiency.

### Dependent Library Installation

Install the following dependent packages using pip commands:
```
# Basic data processing and computation
pip install numpy pandas scipy
# EMD decomposition implementation
pip install PyEMD
# Machine learning model (SVR)
pip install scikit-learn
# Deep learning framework (LSTM)
pip install torch==1.7.1  # Compatible with Python 3.7; adjust the version according to GPU configuration
# Other tools
pip install matplotlib  # Optional, for result visualization
```
### Additional Notes
The EMD implementation relies on the EMD module from the PyEMD library;

The MEMD implementation uses a Python port of the MATLAB version proposed by Rehman and Mandic (2010) (included in the code package, no additional installation required);

Hyperparameter optimization depends on GridSearchCV from scikit-learn; ensure this module is imported correctly.
