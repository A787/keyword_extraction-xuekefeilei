import os
import numpy as np
import pandas as pd
from bert_serving.client import BertClient
from sklearn import preprocessing
from label import dataPrepos

def main():
    dataFile = '' \
               'data/article_journals.csv' \
               ''
    articleData = pd.read_csv(dataFile)
    articleData = articleData.iloc[:100,:]

    VecFile = '' \
               'result/abstract2vec.csv' \
               ''
    vecData = pd.read_csv(VecFile)
    vecData = vecData.iloc[:100,:]

    xkFile = '' \
               'result/xueke.csv' \
               ''
    xkData = pd.read_csv(xkFile)
    # 遍历文件
    ids, titles, lei = [], [], []
    # for i in range(len(articleData['id'])):
    #     abvec = float(vecData['vec'][i])
    #     abvec = np.array(abvec,dtype=float)
    xkvecs=[]
    for j in range(len(xkData['xk1'])):
        text = '%s。%s。%s' % (xkData['xk1'][j], xkData['xk2'][j], xkData['xk3'][j])
        text = dataPrepos(text, 'data/stopWord.txt')
        xkvec = bc.encode(text)
        temp = np.zeros(768)
        for i in range(len(xkvec)):
            temp = temp + xkvec[i]
        temp=preprocessing.scale(temp)
        xkvecs.append(temp)
        ids.append(j)
    result = pd.DataFrame({"id": ids, "vec": xkvecs}, columns=['id', 'vec'])
    result.to_csv("result/vecs/xkvecs.csv", index=False)
if __name__ == '__main__':
    bc = BertClient(ip='10.22.123.49', check_version=False, check_length=False)
    main()