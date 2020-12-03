import pandas as pd
import os
import numpy as np
from bert_serving.client import BertClient
def fenlei(a,b,c):
    words = a["id"] # 词汇
    vecs = a["vec"] # 向量表示
    xclass= b["id"]
    xvec= b.iloc[:,1:]
    pros=[]
    leibie=[]
    for i in range(len(xclass)):
        pro=0.0
        leibie.append(i)
        pro+=np.dot(np.array(a.iloc[c,1:]['vec'][1:-1].split(),dtype=float),np.array(b.iloc[i,1:]['vec'][1:-1].split(),dtype=float))
        pro = pro/(np.linalg.norm(np.array(a.iloc[c,1:]['vec'][1:-1].split(),dtype=float))*np.linalg.norm(np.array(b.iloc[i,1:]['vec'][1:-1].split(),dtype=float)))
        pros.append(pro)
    leibie = pd.DataFrame(leibie, columns=['leibie'])
    proba = pd.DataFrame(pros, columns=['proba'])
    cresult = pd.concat([leibie, proba], axis=1)
    cresult=cresult.sort_values(by="proba" )
    return int(cresult.iloc[391,0:1])
def main():
    dataFile = '' \
               'data/article_journals.csv' \
               ''
    articleData = pd.read_csv(dataFile)

    rootdir = "result/vecs"  # 词向量文件根目录
    data1 = pd.read_csv(os.path.join(rootdir, 'xkvecs.csv'), encoding='utf-8')
    data = pd.read_csv('result/abstract2vec.csv')  # 读取词向量文件数据
    xkdata = pd.read_csv('result/xueke.csv')

    ids, titles, lei = [], [], []

    for i in range(len(data)):
        leibie=fenlei(data,data1,i)
        text = '%s-%s-%s' % (xkdata['xk1'][leibie], xkdata['xk2'][leibie], xkdata['xk3'][leibie])  # 拼接标题和摘要

        article_id = i # 获得文章id
        artile_tit = articleData['title'][article_id] # 获得文章标题
        ids.append(article_id)
        titles.append(artile_tit)
        lei.append(text#.encode("gbk")
                         )
    result = pd.DataFrame({"id": ids, "title": titles, "class": lei}, columns=['id', 'title', 'class'])
    result = result.sort_values(by="id", ascending=True)  # 排序
    result.to_csv("result/class_vec.csv", index=False)
if __name__ == '__main__':
    main()