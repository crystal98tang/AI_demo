# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATA_FILE = '../data/diabetes.csv'
FEAT_COLS = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]


def plot_feat_and_Y(data):
    feat_cols = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
    fig, axes = plt.subplots(2, 5, figsize=(10, 10))
    for i, feat in enumerate(feat_cols):
        data[[feat, "Y"]].plot.scatter(x=feat, y='Y', alpha=0.5,
                                       ax=axes[int(i / 5), i - 5 * int(i / 5)])
    plt.tight_layout()
    plt.savefig('./diabetes.png')
    plt.show()


def main():
    """
        主函数
    """
    diabetes_data = pd.read_csv(DATA_FILE)
    plot_feat_and_Y(diabetes_data)
    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=1 / 5)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    r2_socre = lr_model.score(X_test, y_test)
    print("R2值：", r2_socre)


if __name__ == '__main__':
    main()
