# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


def do_eda_plot_for_iris(iris_data):
    """
    EDA 探索性数据分析
    """
    category_color_dict = {
        'Iris-setosa':      'red',
        'Iris-versicolor':  'blue',
        'Iris-virginica':   'green'
    }

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    for category_name, category_color in category_color_dict.items():

        #
        iris_data[iris_data['Species'] == category_name].plot(ax=axes[0], kind='scatter',
                                                              x='SepalLengthCm', y='SepalWidthCm', label=category_name,
                                                              color=category_color)
        #
        iris_data[iris_data['Species'] == category_name].plot(ax=axes[1], kind='scatter',
                                                              x='PetalLengthCm', y='PetalWidthCm', label=category_name,
                                                              color=category_color)

    axes[0].set_xlabel('Sepal Length')
    axes[0].set_ylabel('Sepal Width')
    axes[0].set_title('Sepal Length vs Sepal Width')

    axes[1].set_xlabel('Petal Length')
    axes[1].set_ylabel('Petal Width')
    axes[1].set_title('Petal Length vs Petal Width')

    plt.tight_layout()
    plt.savefig('./iris_eda.png')
    plt.show()


def do_pair_plot_for_iris(iris_data):
    """
        瀵归涪灏捐姳鏁版嵁闆嗙殑鏍锋湰鐗瑰緛鍏崇郴杩涜鍙鍖�
        鍙傛暟锛�
            - iris_data: 楦㈠熬鑺辨暟鎹泦
    """
    g = sns.pairplot(data=iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']],
                     hue='Species')
    plt.tight_layout()
    plt.show()
    g.savefig('./iris_pairplot.png')


def plot_knn_boundary(knn_model, X, y, fig_title, save_fig):
    """
        绘制分类边界
    """
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(fig_title)

    plt.savefig(save_fig)

    plt.show()


def plot_feat_and_price(house_data):
    """
        绘制房价与其他指标的散点图
    """
    feat_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))     # 2 rows,3 cols
    for i, feat_col in enumerate(feat_cols):
        house_data[[feat_col, 'price']].plot.scatter(x=feat_col, y='price', alpha=0.5,
                                                     ax=axes[int(i / 3), i - 3 * int(i / 3)])
    plt.tight_layout()
    plt.savefig('./house_feat.png')
    plt.show()


def plot_fitting_line(linear_reg_model, X, y, fig_title, save_fig):
    """
        缁樺埗绾挎€ф嫙鍚堟洸绾�
        鍙傛暟锛�
            linear_reg_model:   璁粌濂界殑绾挎€у洖褰掓ā鍨�
            X:                  鏁版嵁闆嗙壒寰�
            y:                  鏁版嵁闆嗘爣绛�
            fig_title:          鍥惧儚鍚嶇О
            save_fig:           淇濆瓨鍥惧儚鐨勮矾寰�
    """
    # 绾挎€у洖褰掓ā鍨嬬殑绯绘暟
    coef = linear_reg_model.coef_

    # 绾挎€у洖褰掓ā鍨嬬殑鎴窛
    intercept = linear_reg_model.intercept_

    # 缁樺埗鏍锋湰鐐�
    plt.scatter(X, y, alpha=0.5)

    # 缁樺埗鎷熷悎绾�
    plt.plot(X, X * coef + intercept, c='red')

    plt.title(fig_title)
    plt.savefig(save_fig)
    plt.show()