import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor

# 得られている情報の数
initial = 50
# 1回の予測に使うペアの個数
n = 10
# 1回の予測に使う株データの個数
m = 5
# 予測の回数
p = 5
# 1回の予測に使うデータの深さ
q = 10

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

# GPR
def GPR_fit(x_train,y_train,x_test):
    kernel = sk_kern.RBF(1.0, (1e-3, 1e3)) + sk_kern.ConstantKernel(1.0, (1e-3, 1e3)) + sk_kern.WhiteKernel()
    clf = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10, 
        optimizer="fmin_l_bfgs_b", 
        n_restarts_optimizer=20,
        normalize_y=True)
    clf.fit(x_train,y_train)
    pred_mean, pred_std = clf.predict(x_test, return_std=True)
    return pred_mean,pred_std

# 期待値を算出する関数
def expectation(x,y):
    y = y / sum(y)
    return sum(x*y)

# kernel density estimationをしたのち，期待値を算出する
def kde_process(data_list):
    kde_model = gaussian_kde(data_list)
    y = kde_model(data_list)
    skew = pd.Series(y).skew()
    if abs(skew) < 0.1:
        return expectation(data_list,y)
    else:
        data_list2=pointsort(data_list,np.average(data_list))[0:int(len(data_list)/3)]
        return expectation(data_list2,kde_model(data_list2))

# データフレームdfの中から，tergetの株のmeanとsdを推定する関数
def onetimeestimation(df,terget,n,m):
    # tergetの情報（1つだけずらして取得)
    y_train = df[terget].values[1:]
    result = np.array([])
    result_sd = np.array([])
    i = 0
    while i < n:
        x = df[pickup2(similar, terget, m)].values
        x_train,x_test = x[:-1],[x[-1]]
        pred_y ,pred_y_sd = GPR_fit(x_train,y_train,x_test)
        result = np.append(result,pred_y)
        result_sd = np.append(result_sd,pred_y_sd)
        i = i+1
        print(i,result)
    mean = kde_process(result)
    sd = np.average(result_sd)
    return mean,sd

# 同業種リストsimilarのすべての1ステップを推定する関数
def onetimeallestimation(df,similar,n,m,q):
    result = []
    result_sd = []
    df = df[-q:]
    for terget in similar:
        result.append(onetimeestimation(df,terget,n,m,)[0])
        result_sd.append(onetimeestimation(df,terget,n,m,)[1])
    return result,result_sd

# 複数回の推定を行う
def manytimeestimation(df,similar,n,m,p,q):
    error = df[0:0].copy()
    for i in range(p):
        mean,sd = onetimeallestimation(df,similar,n,m,q)
        mean = pd.Series(mean,index=similar,name='pred'+str(i))
        sd = pd.Series(sd,index=similar,name=i)
        df = df.append(mean)
        error = error.append(sd)
    return df,error

# グラフの出力のための関数
def makegraph(estimation,sd,real,tergetname):
    x_grid = np.array(range(len(estimation)))
    plt.figure(figsize=(14,7))
    plt.errorbar(x_grid,estimation,sd,fmt='ro-')
    plt.plot(x_grid,real,'go-')
    plt.savefig(tergetname)
    
result, sd = manytimeestimation(df,similar,n,m,p,q)

# result.to_csv('result.csv')
# sd.to_csv('sd.csv')
# result = pd.read_csv('./result.csv')
# sd = pd.read_csv('./sd.csv')

for terget in similar:
    tempestimation = result[terget].values[initial:initial+p]
    tempsd = sd[terget].values[0:p]
    tempreal = df_real[terget].values[initial:initial+p]
    makegraph(tempestimation,tempsd,tempreal,terget)

    
