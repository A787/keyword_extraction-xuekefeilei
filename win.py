import codecs
import os
import sys
from sklearn import preprocessing
import jieba
from PyQt5.QtWidgets import QApplication, QMainWindow
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import main
import pandas as pd
from label import getKeywords_cor
from bert_serving.client import BertClient
bc = BertClient(ip='10.22.123.49', check_version=False, check_length=False)

def key():

    title = ui.plainTextEdit_title.toPlainText()
    abs = ui.plainTextEdit_abs.toPlainText()
    text = '%s。%s' % (title , abs )
    key = jieba.analyse.extract_tags(text, topK=10, withWeight=False, allowPOS=())
    ui.textBrowser_label.setText(','.join(key))
    print(1)

def xk():
    title = ui.plainTextEdit_title.toPlainText()
    abs = ui.plainTextEdit_abs.toPlainText()
    text = '%s。%s' % (title , abs )
    key = jieba.analyse.extract_tags(text, topK=10, withWeight=False, allowPOS=())

    vec = bc.encode(key)
    temp = np.zeros(768)
    for i in range(len(key)):
        temp = temp + vec[i]
    temp = preprocessing.scale(temp)

    xkvec = pd.read_csv('result/vecs/xkvecs.csv', encoding='utf-8')
    xk = pd.read_csv('result/xueke.csv', encoding='utf-8')
    pros = []
    for i in range(len(xkvec)):
        pro = temp.dot(np.array(xkvec.iloc[i,1:]['vec'][1:-1].split(),dtype=float))
        pros.append(pro)
    leibie = pros.index(max(pros))

    xuekefl = '一级：%s\n二级：%s\n三级学科：%s' % (xk['xk1'][leibie], xk['xk2'][leibie], xk['xk3'][leibie])  # 拼接标题和摘要
    ui.textBrowser_xk.setText(xuekefl)
    print(2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = main.Ui_Dialog()
    ui.setupUi(MainWindow)

    ui.pushButton_label.clicked.connect(lambda:key())
    ui.pushButton_xk.clicked.connect(lambda:xk())

    MainWindow.show()
    sys.exit(app.exec_())





