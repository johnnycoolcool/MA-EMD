# MA-EMD: Aligned Empirical Decomposition for Multivariate Time-Series Forecasting
## 引用声明

若使用本代码进行研究、学术发表或任何形式的应用，请务必引用以下论文：Xiangjun Cai, Dagang Li, Jinglin Zhang, Zhuohao Wu. MA-EMD: Aligned empirical decomposition for multivariate time-series forecasting. Expert Systems With Applications, 267(2025): 126080.DOI: https://doi.org/10.1016/j.eswa.2024.126080

## 代码作用

本代码实现了论文提出的MA-EMD（Multivariate Aligned Empirical Mode Decomposition） 方法，用于解决多元时间序列预测中的分解 - 集成模型（DEMs）性能优化问题。

核心功能包括：

1.对多元时间序列的每个变量单独进行标准 EMD 分解，保证分解质量；

2.基于目标变量的本征频率成分，通过 KL 散度实现不同变量分解子序列（IMF）的频率对齐；

3.构建 MA-EMD 基分解集成预测模型，同时提升预测精度和计算效率；

4.支持与 SVR、LSTM 等主流预测器结合，适用于气候、电力、汇率等多领域多元时间序列预测任务。
## 代码输入与输出

### 1. 输入

代码接收结构化的多元时间序列数据，具体要求如下：

数据格式：表格型数据（如 CSV、Excel 文件或 NumPy 数组），其中每一列代表一个独立变量（1 个目标变量 + 多个关联变量），每一行代表一个时间步（按时间顺序排序）。

数据范围：与论文中验证的数据集类型一致，包括但不限于：

    气候数据（如耶拿气候数据集：包含气温、气压、湿度等 14 个变量）；
    电力系统数据（如 ETT 数据集：包含变压器负载、油温等 7 个变量）；
    金融数据（如汇率数据集：包含 8 个国家的日汇率变量）。
    数据预处理要求：原始数据需提前完成预处理（缺失值填充、异常值剔除），代码在执行过程中会自动将数据标准化为零均值、单位方差格式。

关键参数：

    target_var_name：目标变量名称 / 索引（如气候数据中为 “气温”，ETT 数据中为 “油温”）；
    lookback_window：用于预测的历史时间窗口长度（如小时级数据设为 24 以捕捉日趋势，日级数据设为 7 以捕捉周趋势）；
    omega (ω)：部分分解的阈值（控制有效 IMF 所需的最小极值点数量，默认按论文建议设为 50）。

### 2. 输出

代码生成多种类型的输出结果，支持结果分析与实际应用：

中间输出：

    分解后的 IMF（本征模态函数）：字典格式，键为变量名，值为 IMF 构成的 NumPy 数组（形状：[IMF 数量，时间步数量]）；
    对齐后的 IMF 分组：列表形式的子组集合，每个子组包含 1 个目标变量 IMF 及与其对齐的关联变量 IMF（结构与论文表 4 - 表 7 一致）；
    分解与对齐日志：文本文件，记录执行时间（如 EMD 分解时间、KL 散度计算时间），用于计算效率分析。

最终输出：

   经独立分解 + KL 散度对齐后的多元变量 IMF 分组结果，研究者可基于此分组适配任意预测模型。
## 论文核心理论
1. 现有方法痛点
传统多元经验模态分解（MEMD）为实现多变量分解子序列对齐，需放松分解质量约束，导致生成低质量、不必要的频率成分，同时计算开销显著增加，最终影响预测性能。

3. MA-EMD 核心创新

    独立高质量分解：对每个变量单独使用标准 EMD 分解，严格保证本征模态函数（IMF）的周期性和包络对称性，避免 MEMD 的质量损失；

    目标导向对齐机制：以目标变量的 IMF 为参考，通过极值点间隔分布表征 IMF 频率特征，使用 KL 散度度量并匹配其他变量 IMF 的频率成分，实现精准对齐；

    部分分解优化：引入阈值 ω 控制 IMF 有效性，过滤极值点过少的低质量 IMF，平衡分解完整性与预测稳定性。

4. 模型框架
    数据标准化：将多元时间序列转换为零均值、单位方差的格式；

    EMD 分解：对每个变量单独执行 EMD，得到各变量的 IMF 集合；

    频率对齐：基于 KL 散度将相关变量的 IMF 与目标变量 IMF 分组；

    子模型训练：为每个对齐后的 IMF 组训练独立的多元预测子模型；

    结果集成：聚合所有子模型预测结果，得到目标变量最终预测值。

## 安装环境要求

### 基础环境

Python 版本：3.7.3（建议通过 Anaconda 安装）

开发工具：Pycharm 2019.2.3（Community Edition，可选）

硬件支持：建议配备 NVIDIA GPU（如 RTX 3090 Ti）加速模型训练，CPU 可运行但效率较低

### 依赖库安装
通过 pip 命令安装以下依赖包：
```python
# 基础数据处理与计算
pip install numpy pandas scipy
# EMD分解实现
pip install PyEMD
# 机器学习模型（SVR）
pip install scikit-learn
# 深度学习框架（LSTM）
pip install torch==1.7.1  # 适配Python 3.7，可根据GPU配置调整版本
# 其他工具
pip install matplotlib  # 可选，用于结果可视化
```
### 补充说明
EMD 实现依赖PyEMD库的EMD模块；

MEMD 实现采用 Rehman 和 Mandic（2010）提出的 MATLAB 版本移植的 Python 版本（代码包中已包含，无需额外安装）；

超参数优化依赖scikit-learn的GridSearchCV，确保该模块正常导入即可。
