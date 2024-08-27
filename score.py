import pandas as pd
from sklearn.metrics import accuracy_score

# 加载CSV文件
cla_gt = pd.read_csv('./cla_gt.csv')
cla_pre = pd.read_csv('./cla_pre.csv')


cla_gt.columns = ['id', 'true_label']
cla_pre.columns = ['id', 'pred_label']


# 合并结果
cla_merged = pd.merge(cla_gt, cla_pre, on='id')

# 计算分类模型的准确率
acc = accuracy_score(cla_merged['true_label'], cla_merged['pred_label'])


# 打印结果
print(f"鉴别准确率: {acc}")

