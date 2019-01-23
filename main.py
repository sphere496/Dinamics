import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import gaussian_kde

# 得られている情報の数
initial = 28
# 予測に使う数
n = 10
# 予測に使うデータの個数
m = 4
# 予測の回数
p = 30
# 予測に使うデータの深さ
q = 10
# カーネル関数の情報
a = 0.01
b = 0.01
beta = 1000


# 予測する株の名前
terget = '1812 JT Equity'

similar = ['1801 JT Equity','1802 JT Equity','1803 JT Equity','1812 JT Equity','1820 JT Equity','1821 JT Equity','1824 JT Equity','1833 JT Equity','1860 JT Equity','1893 JT Equity']
# 推定のために1/1000倍
df_open = pd.read_csv('./Data/Open.csv',usecols =similar)
df_close = pd.read_csv('./Data/Close.csv',usecols =similar)

df = (df_close - df_open) /df_open
    
df_real = df[0:initial+p]
df = df[0:initial]

# リストを，ある値との距離順に並べる関数
def pointsort(arr,val):
    return [y[1] for y in sorted([(abs(x-val),x) for x in arr])]

# randomにn個の配列を作る
def pickup(arr, n):
    arr2 = arr[:]
    result = []
    for i in range(n):
        x = arr2[int(len(arr2) * np.random.rand())]
        result.append(x)
        arr2.remove(x)
    return result

# aを含まないランダムな配列をn個作る
def pickup2(arr,a,n):
    arr2 = arr[:]
    arr2.remove(a)
    return pickup(arr2,n-1)

# 多変数のグラム行列を作る関数
def gram(f,x):
    result = []
    for i in x:
        smallresult = []
        for j in x:
            smallresult.append(f(i,j))
        result.append(smallresult)
    return np.array(result)

# ベクトルを作る関数
def k(f,new,x):
    result = []
    for i in x:
        result.append(f(new,i))
    return np.array(result)

# カーネル推定のクラス
class GaussianKernel(object):
    def __init__(self, params):
        assert np.shape(params) == (2,)
        self.__params = params

    def __call__(self, x1, x2):
        return self.__params[0] * np.exp(-0.5 * self.__params[1] * np.linalg.norm(x1 - x2) ** 2)

    # 以下の関数はparameterの推定のための用意
    def get_params(self):
        return np.copy(self.__params)

    def derivatives(self, x1, x2):
        delta_1 = -0.5 * sq_diff * delta_0 * self.__params[0]
        return (delta_0, delta_1)

    def delta0(self,x1,x2):
        sq_diff = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-0.5 * self.__params[1] * sq_diff)

    def delta1(self,x1,x2):
        sq_diff = np.linalg.norm(x1 - x2) ** 2
        return -0.5 * sq_diff * self.delta0(x1,x2) * self.__params[0]
            
    def update_parameters(self, updates):
        assert np.shape(updates) == (2,)
        self.__params += updates

# GPR用のクラス 
class GaussianProcessRegression(object):
    def __init__(self, kernel, beta=1.):
        self.kernel = kernel
        self.beta = beta

    def fit(self, x, y):
        self.x = x
        self.y = y
        Gram = gram(self.kernel,x)
        self.covariance = Gram + np.identity(len(x)) / self.beta
        self.precision = np.linalg.inv(self.covariance)

    # Kernel functionのparameter推定 / 今回は綺麗に収束しないので，要検討
    def fit_kernel(self, x, y, learning_rate=0.1, iter_max=100):
        for i in range(iter_max):
            params = self.kernel.get_params()
            self.fit(x, y)
            grad0 = gram(self.kernel.delta0,x)
            grad1 = gram(self.kernel.delta1,x)
            gradients = [grad0,grad1]
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + y.dot(self.precision.dot(grad).dot(self.precision).dot(y)) for grad in gradients])
            print(updates)
            self.kernel.update_parameters(learning_rate * updates)
            if np.allclose(params, self.kernel.get_params()):
                break
        else:
            print("parameters may not have converged")
        
    def predict_dist(self, new):
        K = k(self.kernel, new, self.x)
        mean = K.dot(self.precision).dot(self.y)
        var = self.kernel(new, new) + 1 / self.beta - np.sum(K.dot(self.precision) * K)
        return mean.ravel(), np.sqrt(var.ravel())

# 期待値を算出する関数
def expectation(x,y):
    y = y / sum(y)
    return sum(x*y)

# kernel density estimationをしたのち，期待値を算出する
def kde_process(data_list):
    kde_model = gaussian_kde(data_list)
    y = kde_model(data_list)
    # x_grid = np.linspace(min(data_list), max(data_list), num=100)
    # estimy = kde_model(x_grid)
    # # データを正規化したヒストグラムを表示する用
    # weights = np.ones_like(data_list)/float(len(data_list))
    # # print("mean:", pandasy.mean())
    # plt.figure(figsize=(14,7))
    # plt.plot(x_grid, estimy)
    # plt.hist(data_list, alpha=0.3, bins=20, weights=weights)
    # plt.show()
    skew = pd.Series(y).skew()
    if abs(skew) < 0.1:
        return expectation(data_list,y)
    else:
        data_list2=pointsort(data_list,np.average(data_list))[0:int(len(data_list)/3)]
        return expectation(data_list2,kde_model(data_list2))

# データフレームdfの中から，tergetの株のmeanとsdを推定する関数
def onetimeestimation(df,terget,n,m,a,b,beta):
    # tergetの情報（1つだけずらして取得)
    y = df[terget].values[1:]
    y = y[:n-1]
    kernel = GaussianKernel(params=np.array([a, b]))
    result = np.array([])
    result_sd = np.array([])
    for i in range(n):
        x = df[pickup2(similar, terget, m)].values       
        # 推定前のx(matrix)
        x = x[:n]
        x,x_test = x[:-1],x[-1]
        regression = GaussianProcessRegression(kernel=kernel, beta=beta)
        # regression.fit_kernel(x, y, learning_rate=0.1, iter_max=10000)
        regression.fit(x,y)        
        pred_y, pred_y_sd = regression.predict_dist(x_test)
        result = np.append(result,pred_y)
        result_sd = np.append(result_sd,pred_y_sd)
    mean = kde_process(result)
    sd = np.average(result_sd)
    return mean,sd

# 同業種リストsimilarのすべての1ステップを推定する関数
def onetimeallestimation(df,similar,n,m,q,a,b,beta):
    result = []
    result_sd = []
    df = df[-q:]
    for terget in similar:
        result.append(onetimeestimation(df,terget,n,m,a,b,beta)[0])
        result_sd.append(onetimeestimation(df,terget,n,m,a,b,beta)[1])
    return result,result_sd

# 複数回の推定を行う
def manytimeestimation(df,similar,n,m,p,q,a,b,beta):
    error = df[0:0].copy()
    for i in range(p):
        mean,sd = onetimeallestimation(df,similar,n,m,q,a,b,beta)
        mean = pd.Series(mean,index=similar,name='pred'+str(i))
        sd = pd.Series(sd,index=similar,name=i)
        df = df.append(mean)
        error = error.append(sd)
    return df,error

# グラフの出力のための関数
def makegraph(estimation,sd,real):
    x_grid = np.array(range(len(estimation)))
    plt.figure(figsize=(14,7))
    plt.errorbar(x_grid,estimation,sd,fmt='ro-')
    plt.plot(x_grid,real,'go-')
    plt.show()
    
result, sd = manytimeestimation(df,similar,n,m,p,q,a,b,beta)
estimation = result[terget].values[initial:initial+p]
sd = sd[terget].values[0:p]
real = df_real[terget].values[initial:initial+p]

# # print(onetimeestimation(df,terget,n,m,a,b,beta))
# # print(onetimeallestimation(df,similar,10,3,10))
makegraph(estimation,sd,real)

    
