import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import optimize

np.set_printoptions(suppress=True, threshold=np.nan)  # 去除科学计数法，不然看起来太难受
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'  # mac环境下防止中文乱码

'''
1.logisticRegression()函数调用loadFile函数，载入文件
2.创建两个矩阵X和y。一个用于接收横坐标和纵坐标，一个用于接受实际值(1或0）
3.利用plotData(),先看看整个数据的分布情况。此时红色代表不合格的芯片，蓝色代表合格的芯片。
4.紧接着我们需要调用一个mapFeature()函数。也就是映像为多项式的函数，什么意思呢，因为整个数据就两个特征。打个比方，如果还是按照1+x1+x2一次方程，去模拟逼近，肯定是欠拟合的。
所以我们可以用1+x1+x2+x1^2+x1*x2+x2^2，这样整个图像就会圆滑，并且逐渐逼近原数据。
5.初始化theta值。
6.定义正则化系数lambda的值initial_lambda
7.调用代价函数costFunction()
8.构建梯度函数，这里我们不需要构建梯度下降算法，第九步我们会用fmin_bfgs算法求最优解
9.调用下scipy中的优化算法fmin_bfgs（拟牛顿法Broyden-Fletcher-Goldfar）
    - costFunction是自己实现的一个求代价的函数，                      
    - theta表示初始化的值,                            
    - fprime指定costFunction的梯度                          
    - args是其余测参数，以元组的形式传入，最后会将最小化costFunction的theta返回  
10.构建并调用预测函数predict()。一共有118条数据，不是每一条数据都符合我们算出的模型。预测函数就是为了算出有多少数据符合模型，算出比例。
11.利用plotDecisionBoundary()函数画出决策边界。
'''


# 逻辑回归函数
def logisticRegression():
    data = loadFile('data2.txt')  # 1.载入文件

    X = np.array(data[:, 0:-1])  # 2创建矩阵X和y
    y = np.array(data[:, -1])

    plotData(X, y)  # 3.画图

    X = mapFeature(X[:, 0], X[:, 1])  # 4.映射过的多项式覆盖原来的X，此时的X已经有六列。

    theta = np.zeros((X.shape[1], 1))  # 5.初始化theta值，此时是6行1列的矩阵
    initial_lambda = 0.1  # 6.正则化系数的值lambda一般取0.01,0.1,1 。

    J = costFunction(theta, X, y, initial_lambda)  # 7.调用代价函数
    print(J)  # 测试下J，初始值应该是0.69314718

    result = optimize.fmin_bfgs(costFunction, theta, fprime=gradient,
                                args=(X, y, initial_lambda))  # 9.调用fmin_bfgs算法。返回值返回的是theta的最优解
    p = predict(X, result)  # 10.调用预测函数。result参数为我们在上一步中求出的theta最优解
    print('theta的最优解为:', result)
    print('在当前数据集上，训练的准确度为%f%%' % np.mean(
        np.float64(p == y) * 100))  # p==y实际上是一个bool判断。返回的是一个n行1列的数组，值为False和True,用np.float64转化为0和1数组。

    X = data[:, 0:-1]  # 这里需要注意下，我们要重新把X重新定义下，变成只有两个特征的数组。原来的X因为进行了多项式映射，已经有6个了。
    plotDecisionBoundary(result, X, y)  # 11.画出决策边界。我们需要把theta最优解result代入


# 载入文件函数
def loadFile(path):
    return np.loadtxt(path, delimiter=',', dtype=np.float64)  # 返回一个np对象。其中分隔符为逗号，类型为np.float64类型


# 反应数据分布的画图函数
def plotData(X, y):
    y0 = (y == 0)  # 此时y0存放的是一个数组，数组里面都是bool值,当y中的值为0，则数组中当前的值为true,反之为false。相当于记录了当前y=0在数组中的行位置。
    y1 = (y == 1)

    # y0=np.where(y==0)   #如果你看不懂以上的式子，可以用这段注释定义。此时我们利用where函数寻找y=0的行，此时返回的就是行索引值
    # y1=np.where(y==1)

    plt.figure(figsize=(6, 6))
    plt.plot(X[y0, 0], X[y0, 1], 'b+')  # 因为y0记录了所有的y=0的行位置，现在我只需要把X中当前行的横纵坐标打出来。
    plt.plot(X[y1, 0], X[y1, 1], 'ro')

    plt.title('经过两种测试的芯片数据')
    plt.xlabel('test1')
    plt.ylabel('test2')
    plt.show()


# 映射多项式
def mapFeature(X1, X2):
    X_new = np.ones(
        (X1.shape[0], 1))  # 先初始化一个X_new数组，这个数组最后要返回并且取代X。为什么用1数组呢。根据视频内容，所有数据的theta0对应的x0的一直为1，所以我们现在提前把第一列全部为。

    '''
    我们需要将整个数据映射为1,x1,x1^2,x1*x2,x2^2,其中映射为1我们已经在上一行中做过了
    '''

    degree = 2  # 映射的最高次方。意思是最后映射为：x1,x1^2,x1*x2,x2^2

    for i in range(1, degree + 1):  # 这个双层循环需要自己在草稿纸写一下，才能理解，最后的循环结果就是将x1,x1^2,x1*x2,x2^2放入数组
        for j in range(0, i + 1):
            temp = X1 ** (i - j) * X2 ** j
            X_new = np.hstack((X_new, temp.reshape(-1, 1)))

    return X_new


# 代价函数
def costFunction(theta, X, y, initial_lambda):
    m = len(y)  # 先算出有m条数据
    J = 0  # 初始化代价函数为0

    z = np.dot(X, theta)  # theta*X,返回一个m行1列的数据。
    hx = sigmoid(z)  # 将theta*X代入都sigmoid函数，求出假设函数

    theta1 = theta.copy()
    theta1[0] = 0  # 因为我们要正则化，后面需要加入正则项。正则项中的theta0是不需要被考虑进去的，所以我们需要拷贝一下theta，赋值给一个新的数组theta1，并且将theta1的第一项设为0

    regularizationTerm = initial_lambda * np.dot(np.transpose(theta1), theta1)  # 正则项
    J = (-np.dot(np.transpose(y), np.log(hx)) - np.dot(np.transpose(1 - y),
                                                       np.log(1 - hx)) + regularizationTerm / 2) / m  # 代价函数值
    return J


# S型函数
def sigmoid(z):
    h = np.zeros((len(z), 1))  # 初始化数组
    h = 1.0 / (1.0 + np.exp(-z))

    return h


# 梯度
def gradient(theta, X, y, initial_lambda):
    m = len(y)
    grad = np.zeros((theta.shape[0]))
    hx = sigmoid(np.dot(X, theta))  # 将theta*X代入都sigmoid函数，求出假设函数
    theta1 = theta.copy()
    theta1[0] = 0  # 因为我们要正则化，后面需要加入正则项。正则项中的theta0是不需要被考虑进去的，所以我们需要拷贝一下theta，赋值给一个新的数组theta1，并且将theta1的第一项设为0

    grad = np.dot(np.transpose(X), hx - y) / m + initial_lambda / m * theta1

    return grad


# 预测概率
def predict(X, result):
    m = X.shape[0]

    p = sigmoid(np.dot(X, result))  # 将X*theta代入sigmoid函数，构成假设函数的解p。此时的result为一维数组，进行dot运算会默认转置过的n行1列的数组

    for i in range(m):  # 如果假设函数的值大于0.5,则y值为1.反之为0.
        if p[i] > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p  # 返回的数组是个n行1列的数组。


# 画出决策边界
def plotDecisionBoundary(theta, X, y):
    y0 = (y == 0)
    y1 = (y == 1)
    plt.figure(figsize=(6, 6))
    plt.plot(X[y0, 0], X[y0, 1], 'b+')
    plt.plot(X[y1, 0], X[y1, 1], 'ro')  # 这几步和plotData函数一样，就是将原来数据分布展现出来

    plt.title('决策边界')

    u = np.linspace(-1, 1.5, 50)  # 构建由-1到1.5的50个值得等差数列
    v = np.linspace(-1, 1.5, 50)  # u和v的意思就是横纵坐标，(u,v)代表一个坐标点

    z = np.zeros((len(u), len(v)))  # 构建一个uxv大小的零数组，用来存放每个坐标点上，经过代价函数计算得出的代价值

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = sigmoid(np.dot(mapFeature(u[i].reshape(1, -1), v[j].reshape(1, -1)),
                                     theta))  # 我们要对整个平面上的每个点都进行sigmond的计算，其中theta为最优解求出的result

    z = np.transpose(z)  # 这里必须转置一下。因为contour函数规定，Z : array-like(N, M)，横纵坐标相反
    plt.contour(u, v, z, [0, 0.5])  # 这里得搞清楚，我取[0,0.5]这个区间的值画等高线，因为h(x)在0到0.5之间对应的是芯片无法过检。在这个圈子内都是大于0.5的。
    plt.show()


logisticRegression()



