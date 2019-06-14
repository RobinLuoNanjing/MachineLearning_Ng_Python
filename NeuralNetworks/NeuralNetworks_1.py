import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.io as spio
from scipy import optimize


np.set_printoptions(suppress=True, threshold=np.nan)    #去除科学计数法，不然看起来太难受
matplotlib.rcParams['font.family']='Arial Unicode MS' #mac环境下防止中文乱码


'''
1.导入文件loadMatFile(),注意，跟之前不同，这次导入的文件是.mat文件
2.利用show_data()函数，我们先显示100个数字看看。
3.直接调用predict函数，并且代入通过神经网络运算求好的theta1和theta2
'''



def logisticRegression_oneVsAll():
    data=loadMatFile('ex3data1.mat')    #1.导入文件。在.mat文件中存放的是两个矩阵，X是图片矩阵(5000x400),每一行都是一个数字图片的矩阵
    X=data['X']
    y=data['y']

    m=len(X)

    #我们先选100个数字看看。
    rand_indices=[np.random.randint(0,m) for x in range(100)]   #2.显示100个数字：这一步是利用列表表达式选取100个随机的数字。
    show_data(X[rand_indices, :])  # 显示100个数字

    X=np.hstack((np.ones((len(y),1)),X))     #先将X中补上一列1。
    # X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    data=loadWeight('ex3weights.mat')      #载入已经求好的权重。
    theta1=data['Theta1']
    theta2=data['Theta2']

    theta1=np.transpose(theta1)            #两个权重都需要进行转置才可以运算
    theta2=np.transpose(theta2)


    theta1=np.insert(theta1,0,1,axis=1)    #这里需要注意下，theta1必须添加一个bias偏置


    #这段代码可以不用看，这是我参考别人的代码
    # a1=X                      #5000x401
    # z2=a1@theta1    #5000x26
    #
    # a2=sigmoid(z2)            #5000x26
    # z3=a2@theta2              #5000x10
    #
    # a3=sigmoid(z3)
    #
    # y_pred = np.argmax(a3, axis=1)+1
    #
    # y=y.flatten()      #这里一定要把y转变为1维数组。因为y_pred就是一维数组
    #
    # accuracy = np.mean(y_pred == y)
    # print('accuracy = {0}%'.format(accuracy * 100))



    predict(theta1,theta2,X,y)     #4.调用predict函数。




#导入mat文件
def loadMatFile(path):
    return spio.loadmat(path)       #这里我们需要借助scipy.io的loadmat方法来导入.mat文件


#导入weight
def loadWeight(path):
    return spio.loadmat(path)




# 显示随机的100个数字
'''
显示100个数（若是一个一个绘制将会非常慢，可以将要画的数字整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
'''
def show_data(imgs):
    pad=1      #因为我们显示的是100张图片矩阵的集合，所以每张图片我们可以设置一个分割线，对图片进行划分。pad指的是分割线宽度。
    show_imgs=-np.ones((pad+10*(20+pad),pad+10*(20+pad)))    #初始化一个211x211的矩阵，因为100张图片都是20x20，加上分割线，总共的大小就是211x211。
                                                            #这里需要了解下，如果初始化的矩阵值为-1，则分割线的颜色会是黑色。

    row=0    #因为我们要显示100个数字，所以我们需要从图片数组的第0行遍历到第99行，这个row是用来控制遍历的行数

    for i in range(10):     #双层循环，100张图片放进去。
        for j in range(10):
            show_imgs[pad+i*(20+pad):pad+i*(20+pad)+20,pad+j*(20+pad):pad+j*(20+pad)+20]=(     #这段代码比较复杂。等号左边是从show_imgs这个大矩阵中给图片挑选位置。需要注意图片与图片之间都需要留位置给分割线
                imgs[row,:].reshape(20,20,order='F'))      #因为imgs中的每个数字是一行400个像素数据，我们需要将其改造为20x20的矩阵。order=F,是指列优先对原数组进行reshape。因为python默认的是以行优先，但是matlab是列优先。如果不加这个的话，所有的数字都是横着显示的
            row+=1

    plt.imshow(show_imgs,cmap='gray')  # 显示灰度图像,plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。其后跟着plt.show()才能显示出来。
    plt.axis('off')  #把显示的轴去掉
    plt.show()



#Sigmoid函数
def sigmoid(z):
    hx=np.ones((len(z),1))    #初始化一列数组，里面用于存放经过S函数变换后得值。
    hx=1.0/(1.0+np.exp(-z))

    return hx



#预测函数
def predict(theta1,theta2,X,y):
    z=np.dot(X,theta1)     #此时是5000x26
    layer2=sigmoid(z)      #将theta代入到假设函数中去

    z2=np.dot(layer2,theta2)   #此时是5000x10
    hx=sigmoid(z2)

    m=X.shape[0]

    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    p=np.where(hx[0,:]==np.max(hx,axis=1)[0])           #我们需要知道每一行中的最大值，因为这个最大值对应的列数就是我们预测的数字。
    for i in range(1,m):                                #打个比方，第一行数据对应的实际值是0，那么如果预测准确，这一行中的最大值应该是在第0列。
        temp=np.where(hx[i,:]==np.max(hx,axis=1)[i])
        p=np.vstack((p,temp))                           #我们将每一行中的最大值对应的列都添加到p数组中，此时p数组就是存放的每个数字的预测值。

    p=p+1             #这里需要将p加1，因为在这个数据集中，0对应的y值是10。所以会导致hx发生偏移。什么意思呢？我们是利用已经求好的theta1和theta2进行计算。求出的
                    #结果hx实际上是按照[1,2,3,4,5,6,7,8,9,0]排列的。而他们所对应的索引是[0,1,2,3,4,5,6,7,8,9]。也就是说我们通过循环求出的索引实际上是偏小的。得+1

    print('在当前数据集上，训练的准确度为%f%%'%np.mean(np.float64(p==y)*100))   #我们将预测值p与实际值y，进行比较。如果相同，则为True，不同则为False，通过np.float计算，True为1，Flase为0.
    return p



#调用 logisticRegression_oneVsAll函数
logisticRegression_oneVsAll()