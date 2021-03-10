# -*- coding: utf-8 -*-
"""
    mission: K值的选用
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ai_utils import plot_knn_boundary

DATA_FILE = '../data/Iris.csv'
SPECIES_LABEL_DICT = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
#
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def investigate_knn(iris_data, sel_cols, k_val):
    """
       the different k values influence on model
    """
    X = iris_data[sel_cols].values
    y = iris_data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)

    knn_model = KNeighborsClassifier(n_neighbors=k_val)

    knn_model.fit(X_train, y_train)
    acc = knn_model.score(X_test, y_test)
    print('k={}, accuracy={:.2f}%'.format(k_val, acc * 100))
    plot_knn_boundary(knn_model, X_test, y_test,
                      'PetalLengthCm vs PetalWidthCm, k={}'.format(k_val),
                      save_fig='Petal_k={}.png'.format(k_val))


def main():
    """
    主函数
    :return:
    """
    # read dataset
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')
    iris_data['label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    #
    k_vals = [3, 5, 10]
    sel_cols = ['PetalLengthCm', 'PetalWidthCm']
    for k_val in k_vals:
        print('k={}'.format(k_val))
        investigate_knn(iris_data, sel_cols, k_val)


if __name__ == '__main__':
    main()
