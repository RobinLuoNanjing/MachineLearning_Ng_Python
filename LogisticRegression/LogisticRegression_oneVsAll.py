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
3.调用oneVsAll()函数。因为我们利用的是optimize中的bfgs算法，所以我们需要提前写出以下的函数：
    -sigmoid函数
    -costFunction函数
    -gradient函数
4.调用predict函数，看看在当前数据样本中，我们算出的假设函数的准确性是多少
'''



def logisticRegression_oneVsAll():
    data=loadMatFile('data_digits.mat')    #1.导入文件。在.mat文件中存放的是两个矩阵，X是图片矩阵(5000x400),每一行都是一个数字图片的矩阵
    X=data['X']
    y=data['y']
    m=len(X)

    #我们先选100个数字看看。
    rand_indices=[np.random.randint(0,m) for x in range(100)]   #2.显示100个数字：这一步是利用列表表达式选取100个随机的数字。
    show_data(X[rand_indices, :])  # 显示100个数字

    Lambda=0.1 #选定一个合适的正则化系数。

    num_labels=10   #这里对0-9数字做个标记，用于后面对数字的遍历。


    X=np.hstack((np.ones((len(y),1)),X))     #先将X中补上一列1。

    all_theta=oneVsAll(X,y,num_labels,Lambda)    #3.调用oneVsAll()函数。

    predict(all_theta,X,y)     #4.调用predict函数。




#导入mat文件
def loadMatFile(path):
    return spio.loadmat(path)       #这里我们需要借助scipy.io的loadmat方法来导入.mat文件





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




#oneVsAll函数
def oneVsAll(X,y,num_labels,initial_lambda):
    m,n=X.shape     #m表示X的行数，n表示X的列数，此时的m应该等于5000，列数等于501。

    all_theta=np.zeros((n,num_labels))    #初始化theta值。同时用来存放用于预测0-9每个数字的theta值。也就是说，分类数字1的时候，我们需要一组theta，分类数字2的时候，我们也需要一组theta。

    class_y=np.zeros((m,num_labels))      #这个数组，用于存放(0/1)值。这一步比较难理解，直接跳到下面步骤映射y，可以进一步理解。

    initial_theta=np.zeros((n,1))        #设置初始的theta值。全为0

    #映射y
    for i in range(num_labels):        #class_y一共有10列，每一列都是用于预测一个数字，比如第0列就是预测0，第一列就是预测1。
        class_y[:,i]=np.int32(y==i).reshape(1,-1)   #当i遍历到第0列时，意味着，我们需要对0进行分类。此时的所有"0"数字，他的y值就是1，其他值都是0。以此类推，如果遍历到第一列，那么除1以外的其他行对应的y值都为0.

    #进行分类
    for i in range(num_labels):         #分别对0-9的数字进行分类。这里依旧采用的是上一作业提到的bfgs算法。
        result=optimize.fmin_bfgs(costFunction,initial_theta,fprime=gradient,args=(X,class_y[:,i],initial_lambda))
        all_theta[:,i]=result.reshape(1,-1)

    return all_theta


#求梯度
def gradient(theta,X,y,initial_lambda):     #梯度就是对每个参数进行偏导。这里参数指的就是theta。
    m=len(y)
    grad=np.zeros((theta.shape[0]))   #初始化grad数组。

    hx=sigmoid(np.dot(X,theta))    #调用sigmoid函数

    theta1=theta.copy()    #因为正则化的时候，默认第一个theta值不需要被正则化，所以我们拷贝一个theta作为theta1
    theta1[0]=0            #同时将theta1的第一位改为0


    grad=np.dot(np.transpose(X),hx-y)/m+initial_lambda*theta1/m


    return grad




#损失函数
def costFunction(theta,X,y,initial_lambda):
    m=len(y)
    J=0

    z=np.dot(X,theta)     #X*theta,生成一个m行，1列的数组。
    h=sigmoid(z)          #代入sigmoid函数

    theta1=theta.copy()    #因为正则化的时候，默认第一个theta值不需要被正则化，所以我们拷贝一个theta作为theta1
    theta1[0]=0            #同时将theta1的第一位改为0

    J=-(np.dot(np.transpose(y),np.log(h))+np.dot(np.transpose(1-y),np.log(1-h)))/m +(initial_lambda*np.dot(np.transpose(theta1),theta1))/(2*m)


    return J


#Sigmoid函数
def sigmoid(z):
    hx=np.ones((len(z),1))    #初始化一列数组，里面用于存放经过S函数变换后得值。
    hx=1.0/(1.0+np.exp(-z))

    return hx


#预测函数
def predict(all_theta,X,y):
    z=np.dot(X,all_theta)
    hx=sigmoid(z)      #将theta代入到假设函数中去

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

    print('在当前数据集上，训练的准确度为%f%%'%np.mean(np.float64(p==y)*100))   #我们将预测值p与实际值y，进行比较。如果相同，则为True，不同则为False，通过np.float计算，True为1，Flase为0.
    return p




#调用 logisticRegression_oneVsAll函数
logisticRegression_oneVsAll()