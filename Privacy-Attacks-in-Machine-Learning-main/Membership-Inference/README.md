# Membership Inference Attack

## Overview

这是对Membership inference attack [1] 的简单实现, 假设攻击者已知目标模型的框架结构.只训练了一个影子模型,影子数据集为原数据集添加高斯噪声.目标模型是神经网络,由三组Linear、Tanh组成;攻击模型也是神经网络,由Linear、ReLU、Linear构成.

## Dataset
purchase数据集

https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz

## Requirements

    |-Python 3.11

    |-Pytorch 2.0.1


## Project structure:

    Membership_Inference 
     |- best_models (保存训练效果最好的目标模型和影子模型)
        |-best_target_model.pth (最好的目标模型)
        |-best_shadow_model.pth (最好的影子模型)
     |- ShadowAttack.py (主文件 用以执行推断攻击,包括训练目标模型与影子模型)
     |- model.py  (目标模型、影子模型、攻击模型的结构)
     |- prepare_purchase.py  (准备数据)
     |- data_partition.py  (数据分割)
     |- env.yml (数据集路径)
     |- README.md

    root_dir
     |-purchase （分割后的数据）
     |-tmp (下载的数据)



## Run Experiements:

首先下载数据集并进行分割
```
  python prepare_purchase.py
  
  python data_partition.py
```
之后进行推断攻击
```
  python ShadowAttack.py  
```





## References:
1. Shokri, R., Stronati, M., Song, C. and Shmatikov, V., 2017, May. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy (SP) (pp. 3-18). IEEE.
2. https://github.com/inspire-group/MIAdefenseSELENA


