import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = '../data/fruit_data.csv'
FRUIT_DICT = {
    'apple':    0,
    'lemon':    1,
    'mandarin': 2,
    'orange':   3
}

FEAT = ['mass', 'width', 'height', 'color_score']


def main():
    data = pd.read_csv(DATA_FILE)
    data['label'] = data['fruit_name'].map(FRUIT_DICT)
    #
    X = data[FEAT].values
    y = data['label'].values
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=1 / 5)
    #
    print('原始数据集共{}个样本，其中训练集样本数为{}，测试集样本数为{}'.format(
        X.shape[0], X_train.shape[0], X_test.shape[0]))
    #
    knn_model = KNeighborsClassifier()
    #
    knn_model.fit(X_train, y_train)
    #
    acc = knn_model.score(X_test, y_test)
    #
    print("acc:{:.2f}%".format(acc * 100))


if __name__ == '__main__':
    main()
