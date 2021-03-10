# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ai_utils import plot_feat_and_price

DATA_FILE = '../data/house_data.csv'
FEAT_COLS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", ]


def main():
    house_data = pd.read_csv(DATA_FILE, usecols=FEAT_COLS + ['price'])
    plot_feat_and_price(house_data)
    X = house_data[FEAT_COLS].values
    y = house_data['price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=1 / 3)

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)
    r2_score = lr_model.score(X_test, y_test)
    print('模型的R2值', r2_score)

    # single
    i = 20
    single_test_feat = X_test[i,:]
    y_true = y_test[i]
    y_pred = lr_model.predict([single_test_feat])
    print("样本特征", single_test_feat)
    print("真实价格：{}，预测价格：{}".format(y_true, y_pred))


if __name__ == '__main__':
    main()
