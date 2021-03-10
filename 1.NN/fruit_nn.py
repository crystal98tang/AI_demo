from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CSV_FILE = '../data/fruit_data.csv'
FEAT = ['mass', 'width', 'height', 'color_score']


def eda(data):
    fruit_COLER_dict = {
        'red': 'apple',
        'orange': 'mandarin',
        'yellow': 'orange',
        'green': 'lemon'
    }
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for color, name in fruit_COLER_dict.items():
        #
        data[data['fruit_name'] == name].plot(ax=axes[0], kind='scatter',
                                              x='width', y='height', label=name,
                                              color=color)
        data[data['fruit_name'] == name].plot(ax=axes[1], kind='scatter',
                                              x='mass', y='color_score', label=name,
                                              color=color)
    axes[0].set_xlabel('width')
    axes[0].set_ylabel('height')
    axes[0].set_title('width vs height')

    axes[1].set_xlabel('mass')
    axes[1].set_ylabel('color_score')
    axes[1].set_title('mass vs color_score')

    plt.tight_layout()
    plt.savefig('./fruit_eda.png')
    plt.show()


def get_predict_label(test_feat, train_data):
    dis_list = []
    for idx, row in train_data.iterrows():
        train_feat = row[FEAT].values
        dis = euclidean(test_feat, train_feat)
        dis_list.append(dis)
    #
    pos = np.argmin(dis_list)
    return train_data.iloc[pos]['fruit_name']


def main():
    """
    main function
    """
    data = pd.read_csv(CSV_FILE)
    eda(data)
    train_data, test_data = train_test_split(data, test_size=1 / 5, random_state=20)
    #
    acc_count = 0
    #
    for idx, row in test_data.iterrows():
        test_feat = row[FEAT].values
        pred_label = get_predict_label(test_feat, train_data)
        true_label = row['fruit_name']
        print("水果{}被预测为{}".format(true_label, pred_label))
        if pred_label == true_label:
            acc_count += 1
    #
    accuracy = acc_count / test_data.shape[0]
    print("准确率:{:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    main()
