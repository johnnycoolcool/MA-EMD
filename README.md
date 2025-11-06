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
