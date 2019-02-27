# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:52:54 2019

@author: maeta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
clf = linear_model.LinearRegression()

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

#予測期間
start = 20160104
end = 20161230

similar = ['1801 JT Equity','1802 JT Equity','1803 JT Equity','1812 JT Equity','1820 JT Equity','1821 JT Equity','1824 JT Equity','1833 JT Equity','1860 JT Equity','1893 JT Equity']
# 推定のために1日リターンに
df_open = pd.read_csv('Open.csv',usecols =similar)
df_close = pd.read_csv('Close.csv',usecols =similar)
df = (df_close - df_open) /df_open

date = pd.read_csv('Date.csv')
po_start = int(date[date['Date']==start]['No'])
po_end = int(date[date['Date']==end]['No'])
index = date['Index'][po_start:po_end+1]

##GPRによる方法
# dfのsimilarに入っている株を，p個ペアでq回ずつ，nの深さで予想する．
class GPRestimation:
    def __init__(self, df, similar, n, p, q):
        self.df = df
        self.similar = similar
        self.n = n
        self.p = p
        self.q = q

    # リストを，ある値との距離順に並べる関数
    def pointsort(self,arr,val):
        return [y[1] for y in sorted([(abs(x-val),x) for x in arr])]

    # randomにn個の配列を作る
    def pickup(self,arr,n):
        arr2 = arr[:]
        result = []
        for i in range(n):
            x = arr2[int(len(arr2) * np.random.rand())]
            result.append(x)
            arr2.remove(x)
        return result

    # aを含まないランダムな配列をn個作る
    def pickup2(self,arr,a,n):
        arr2 = arr[:]
        arr2.remove(a)
        return self.pickup(arr2,n-1)

    # GPR
    def GPR_fit(self,x_train,y_train,x_test):
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
    def expectation(self,x,y):
        y = y / sum(y)
        return sum(x*y)

    # kernel density estimationをしたのち，期待値を算出する
    def kde_process(self,data_list):
        kde_model = gaussian_kde(data_list)
        y = kde_model(data_list)
        skew = pd.Series(y).skew()
        if abs(skew) < 0.1:
            return self.expectation(data_list,y)
        else:
            data_list2=self.pointsort(data_list,np.average(data_list))[0:int(len(data_list)/3)]
            return self.expectation(data_list2,kde_model(data_list2))

    def onetimeestimation(self,i,terget):
        # tergetの情報（1つだけずらして取得)
        y_train = self.df[terget].values[i-self.n:i]
        result = np.array([])
        result_sd = np.array([])
        j = 0
        while j < self.q:
            x = self.df[self.pickup2(self.similar, terget, self.p)].values
            x_train,x_test = x[i-self.n:i],[x[i]]
            pred_y ,pred_y_sd = self.GPR_fit(x_train,y_train,x_test)
            result = np.append(result,pred_y)
            result_sd = np.append(result_sd,pred_y_sd)
            j += 1
        mean = self.kde_process(result)
        sd = np.average(result_sd)
        return mean,sd

    # 同業種リストsimilarのすべての1ステップを推定する関数
    def onetimeallestimation(self,i):
        result = []
        result_sd = []
        for terget in self.similar:
            result.append(self.onetimeestimation(i,terget)[0])
            result_sd.append(self.onetimeestimation(i,terget)[1])
        return result,result_sd

## 線形回帰による方法
class Linearestimation:
    def __init__(self, df, similar, n):
        self.df=df
        self.similar = similar
        self.n = n
        
    def linearregression(self,train_x,train_y,test_x):
        clf.fit(train_x,train_y)
        return np.dot(clf.coef_,test_x)

    def linearestimation(self,terget,n,i):
        train = self.similar[:]
        train.remove(terget)
        train_x = self.df[train].values[i-n:i]
        train_y = self.df[terget].values[i-n:i]
        test_x = self.df[train].values[i:i+1][0]
        return(self.linearregression(train_x,train_y,test_x))
    
    def linearonetimeallestimation(self,i):
        result = np.array([])
        for terget in similar:
            result = np.append(result,self.linearestimation(terget,n,i))
        return result


# PortRet = []
# GPR = GPRestimation(df,similar,10,3,5)
# for i in range(po_start,po_end+1):
#     tmpdf = df[i:i+1]
#     result,result_sd = GPR.onetimeallestimation(i)
#     long = similar[np.argmax(result)]
#     short = similar[np.argmin(result)]
#     portreturn = tmpdf[long].values[-1] - tmpdf[short].values[-1]
#     PortRet.append(portreturn)
#     print(str(i)+"...finished")
# PortRet = pd.DataFrame(PortRet)
# PortRet.index = index
# PortRet.columns = ['Ret']  
# PortRet.to_csv('Result-GPR-test.csv')

# linearcheck
PortRet = []
Line = Linearestimation(df,similar,10)
for i in range(po_start,po_end+1):
    tmpdf = df[i-11:i]
    result = Line.linearonetimeallestimation(i)
    long = similar[np.argmax(result)]
    short = similar[np.argmin(result)]
    portreturn = tmpdf[long].values[-1] - tmpdf[short].values[-1]
    PortRet.append(portreturn)
    print(str(i)+"...finished")
PortRet = pd.DataFrame(PortRet)
PortRet.index = index
PortRet.columns = ['Ret']
PortRet.to_csv('Result-Linear.csv')
