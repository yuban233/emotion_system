from utils import tokenize, load_curpus, load_test, load_my_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn import metrics

'''
analyse.py主程序

需要配置的环境：
sklearn;xgboost;pandas;numpy;re;jieba

验收测试步骤如下：
①修改my_data数据路径
②微调xgboost模型参数
③运行
'''

# 1.加载数据,
data = load_curpus("dataset/train.csv")
my_data = load_test("dataset/测试集输入.csv")  # 【①修改my_data数据路径】
# my_data=load_my_data("dataset/mytest.csv")
# print(data[:10])
# print(my_data[:10])

# 训练
# train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)  # 训练集和测试集8:2
# 测试
train_data = data
test_data = my_data

# train_df = pd.DataFrame(train_data, columns=['review', 'label'])
# test_df = pd.DataFrame(test_data, columns=['review', 'label'])
# print(train_data)
# print(len(train_data))
# print(test_data)
# print(len(test_data))

# 停词表
stopwords = []
with open("stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())

# 2.词袋模型
# data_str = [" ".join(review) for review, label in train_data] + \
#            [" ".join(review) for review, label in test_data]
data_str = [" ".join(review) for review, label in train_data] + \
           [" ".join(review) for review in test_data]
# CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数
vectorizer = CountVectorizer(token_pattern='\[?\w+\]?', stop_words=stopwords, max_features=1000)
vec_fit = vectorizer.fit_transform(data_str)
# print(vec_fit)
# print(vec_fit.toarray())
#
X_data, y_data = [], []
for review, label in train_data:
    # X, y = [], label
    X_data.append(" ".join(review))
    y_data.append(label)
X_train = vectorizer.transform(X_data)
y_train = y_data
# print(X_train)

# X_data, y_data = [], []
# for review, label in test_data:
#     # X, y = [], label
#     X_data.append(" ".join(review))
#     y_data.append(label)
# X_test = vectorizer.transform(X_data)
# y_test = y_data

X_data = []
for review in test_data:
    X_data.append(" ".join(review))
X_test = vectorizer.transform(X_data)

# 3. XGBoost模型  【②微调参数】
xgb = XGBClassifier(n_estimators=200,  # 200棵树
                    early_stopping_rounds=50,  # 当连续n次迭代，分数没有提高后，提前终止训练，防止过拟合50
                    max_depth=5,  # 每棵树的最大深度为10
                    subsample=0.3,  # 样本下采样0.7
                    learning_rate=0.1,  # 学习率0.1
                    scale_pos_weight=0.9)  # 不平衡分类，适当降低正面情感权重有助于提升总体准确率0.5
xgb.fit(X_train, y_train)
result = xgb.predict(X_test)
# print(result)
# print(type(result))
data = pd.DataFrame(result)
save_path = "dataset/predict_result.csv"
with open(save_path, 'w', encoding='utf-8')as f:
    data.to_csv(save_path, index=False, header=None)

# 4.模型评估
# print(metrics.classification_report(y_test, result))
# print("准确率:", metrics.accuracy_score(y_test, result))

# features = vectorizer.get_feature_names()
# weights = xgb.feature_importances_
# for i, index in enumerate(weights.argsort()[::-1][:50]):
#     print(i, ": ", features[index])
