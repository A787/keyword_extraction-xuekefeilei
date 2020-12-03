import pandas as pd
from bert_serving.client import BertClient
from label import dataPrepos
import numpy as np
from sklearn import preprocessing

def main():
    # 读取数据集
    dataFile = '' \
               'result/keys_TFIDF.csv' \
               ''
    data = pd.read_csv(dataFile)
    data=data.iloc[:100, :]
    idList, titleList, keyList = data['id'], data['title'], data['key']
    vecs = []
    corpus = [] # 将所有文档输出到一个list中，一行就是一个文档
    # tf-idf关键词抽取
    for index in range(len(idList)):
        vec = bc.encode(keyList[index].split(' '))
        temp = np.zeros(768)
        for i in range(10):
           temp = temp + vec[i]
        temp = preprocessing.scale(temp)
        vecs.append(temp)
    result = pd.DataFrame({"id": data['id'], "vec": vecs}, columns=['id', 'vec'])
    result.to_csv("result/abstract2vec.csv",index=False)

if __name__ == '__main__':
    bc = BertClient(ip='10.22.123.49', check_version=False, check_length=False)
    main()