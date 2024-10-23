# KAN4Finance

## 使用

`python run.py --datapath="your path of dataset"`

## 文件说明

### arguments.py

设置参数。

### data_handler.py

数据读取及预处理，包括去缺失值、标准化、Winsorize

### models.py

存放各种模型。目前仅包含利用特征直接预测收益率。

### run.py

模型的训练、预测以及可视化。