import jieba
import re
import pandas as pd

'''
配置库utils.py
'''


def tokenize(text):
    # 数据预处理:语料清洗
    text = re.sub("\{%.+?%\}", " ", text)  # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)  # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)  # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    icons = re.findall("\[.+?\]", text)  # 提取出所有表情图标
    text = re.sub("\[.+?\]", "IconMark", text)  # 将文本中的图标替换为`IconMark`

    tokens = []
    for k, w in enumerate(jieba.lcut(text)):  # 精确模式：把文本精确的切分开，不存在冗余单词（就是切分开之后一个不剩的精确组合）
        w = w.strip()
        if "IconMark" in w:  # 将IconMark替换为原图标
            for i in range(w.count("IconMark")):
                tokens.append(icons.pop(0))
        elif w and w != '\u200b' and w.isalpha():  # 只保留有效文本
            tokens.append(w)
    return tokens


def load_curpus(path):
    # 加载语料库
    data = []
    with open(path, "r", encoding="utf-8") as f:
        data = pd.read_csv(f)
        # 打印数据基本信息
        # data = pd.read_csv("dataset/train.csv")
        # print("data shape:", data.shape)
        # print("data example:", data.head(10))
        # print("data describe:", data.describe())
        # count_label = data['label'].value_counts()
        # print("正负类样本比例：\n", count_label)

        # # 少量数据训练
        # data = data.head(10000)
        tok_data = []
        # 对于每一行，通过列名name访问对应的元素
        for index, row in data.iterrows():
            row['review'] = tokenize(row['review'])
            tok_data.append((row['review'], row['label']))
            # print(index)
            # print(row['review'],row['label'])
        # print(tok_data[:10])
    return tok_data


def load_test(path):
    with open(path, "r", encoding="utf-8") as f:
        data = pd.read_csv(f, header=None, names=['review'])
    # print(data)
    # data = pd.DataFrame(data, columns=["review"])
    tok_data = []
    for index, row in data.iterrows():
        # print(index)
        # print(row)
        row['review'] = tokenize((row['review']))
        tok_data.append(row['review'])
    # print(tok_data[:10])
    return tok_data

def load_my_data(path):
    # 加载语料库
    data = []
    with open(path, "r", encoding="utf-8") as f:
        data = pd.read_csv(f)
        tok_data = []
        # 对于每一行，通过列名name访问对应的元素
        for index, row in data.iterrows():
            row['review'] = tokenize(row['review'])
            tok_data.append((row['review']))
            # print(index)
            # print(row['review'],row['label'])
        # print(tok_data[:10])
    return tok_data