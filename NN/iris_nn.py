# -*- coding: utf-8 -*-
"""
    mission: 欧式距离 分类器
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np
import ai_utils

DATA_FILE = '../data/Iris.csv'
SPECIES = ['Iris-setosa',
           'Iris-versicolor',
           'Iris-virginica']
#
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def get_pred_label(test_sample_feat, train_data):
    """
    "欧式距离" 找最近样本
    :param test_sample_feat:
    :param train_data:
    :return:
    """
    dis_list = []
    for idx, row in train_data.iterrows():
        train_sample_feat = row[FEAT_COLS].values
        #
        dis = euclidean(test_sample_feat, train_sample_feat)
        dis_list.append(dis)

    # 最短距离对应的位置
    pos = np.argmin(dis_list)
    return train_data.iloc[pos]['Species']


def main():
    """
    主函数
    :return:
    """
    # read dataset
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')
    # EDA
    ai_utils.do_eda_plot_for_iris(iris_data)
    # dataset split
    train_data, test_data = train_test_split(iris_data, test_size=1 / 3, random_state=10)

    # predict 
    acc_count = 0

    # classifier
    for idx, row in test_data.iterrows():
        # feature
        test_sample_feat = row[FEAT_COLS].values
        # predict
        pred_label = get_pred_label(test_sample_feat, train_data)
        # true
        true_label = row['Species']
        print('样本{}的真标签{},预测标签{}'.format(idx, true_label, pred_label))

        if true_label == pred_label:
            acc_count += 1

    # acc
    accuracy = acc_count / test_data.shape[0]  # 正确数/行数
    print('预测准确率{:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()
