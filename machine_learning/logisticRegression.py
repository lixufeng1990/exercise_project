import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# 绘制原始数据图
def plot_data():
    data = np.loadtxt("./LrTestData/ex2data1.txt", delimiter=",")
    data0 = data[data[:, 2] == 0] #取所有第2列为0的行
    data1 = data[data[:, 2] == 1] #取所有第2列为1的行
    plt.plot(data0[:, 0], data0[:, 1], 'o')
    plt.plot(data1[:, 0], data1[:, 1], '+')
    plt.xlabel("score1")
    plt.ylabel("score2")
    plt.title("two exam score")
    plt.show()


def sigmoid(z):
    return 1/(1 + np.exp(-z))

# 绘制决策边界，
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02): #resolution:分辨率
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)) #绘制坐标网格图
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')
    plt.show()

# 绘制决策边界,数据标准化后,适用于0=theta1*x1+theta2*x2+theta0
def plot_decision_line(x_regular, Y, classifier):
    # 绘制数据图，o是分类1，+分类2
    data_regular = np.hstack((x_regular, Y.reshape(100, 1)))
    data0 = data_regular[data_regular[:, 2] == 0]
    data1 = data_regular[data_regular[:, 2] == 1]
    plt.plot(data0[:, 0], data0[:, 1], 'o')
    plt.plot(data1[:, 0], data1[:, 1], '+')
    # 绘制决策边界
    xx1 = np.arange(np.min(x_regular), np.max(x_regular), 0.01)
    exam2 = -(classifier.coef_[0][0] * xx1 + classifier.intercept_[0]) / classifier.coef_[0][1]
    plt.plot(xx1, exam2)
    plt.xlabel("exam1")
    plt.ylabel("exam2")
    plt.title("regular exam score")
    plt.show()


# 测试集，画图对预测值和实际值进行比较
def test_validate(x_test, y_test, y_predict, classifier):
    x = range(len(y_test))
    plt.plot(x, y_test, "ro", markersize=8, zorder=3, label=u"true_v")
    plt.plot(x, y_predict, "go", markersize=14, zorder=2, label=u"predict_v,$R^2$=%.3f" % classifier.score(x_test, y_test))
    plt.legend(loc="center left")
    plt.xlabel("number")
    plt.ylabel("true?")
    plt.show()


def logistic_regression():
    data = np.loadtxt("./LrTestData/ex2data1.txt", delimiter=",")
    # 提取数据
    x = data[:, 0:2]  #第一个"："表示取所有的行，第二个"0：2"表示取从0列开始到第2列，但是不包括第2列，也就是第0列和第1列
    y = data[:, 2]

    # 对数据的训练集进行标准化
    ss = StandardScaler()
    x_regular = ss.fit_transform(x)
    # 划分训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(x_regular, y, test_size=0.1)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    # 模型效果获取
    r = lr.score(x_train, y_train)
    print("R值(准确率):", r)
    print("theta: ", lr.coef_)
    print("截距(theta0):", lr.intercept_)
    # 预测
    y_predict = lr.predict(x_test)  # 预测

    # 绘制原始数据
    plot_data()
    # 绘制决策边界，自己写的
    plot_decision_line(x_regular, y, classifier=lr)
    # 绘制决策边界，博客找的方法
    plot_decision_regions(x_regular, y, classifier=lr)
    # 绘制测试集结果验证
    test_validate(x_test=x_test, y_test=y_test, y_predict=y_predict, classifier=lr)


logistic_regression()