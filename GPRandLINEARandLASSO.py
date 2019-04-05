# -*- coding: utf-8 -*-
"""
Created on Thu March 7 10:04:54 2019
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

#予測期間
start = 20180104
end = 20181228

similar = ['1801 JT Equity',
'1802 JT Equity',
'1803 JT Equity',
'1812 JT Equity',
'1820 JT Equity',
'1821 JT Equity',
'1824 JT Equity',
'1833 JT Equity',
'1860 JT Equity',
'1893 JT Equity']

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
        print(data_list)
        y = kde_model(data_list)
        skew = pd.Series(y).skew()
        average = self.expectation(data_list,y)
        print(skew)
        if abs(skew) < 0.5:
            return average
        else:
            sortedlist = self.pointsort(data_list,average)[0:int(len(data_list)/2)]
            delta = [x-average for x in sortedlist]
            omega = [np.e**(-x/delta[0]) for x in delta]
            return self.expectation(sortedlist,omega)

    def onetimeestimation(self,i,terget):
        # tergetの情報（1つだけずらして取得)
        y_train = self.df[terget].values[i-self.n:i]
        result = np.array([])
        result_sd = np.array([])
        j = 0
        while j < self.q:
            try:
                x = self.df[self.pickup2(self.similar, terget, self.p)].values
                x_train,x_test = x[i-self.n:i],[x[i]]
                pred_y ,pred_y_sd = self.GPR_fit(x_train,y_train,x_test)
                result = np.append(result,pred_y)
                result_sd = np.append(result_sd,pred_y_sd)
                j += 1
            except:
                j += 1
        mean = self.kde_process(result)
        err = (self.df[terget].values[i+1]-mean)**2
        sd = np.average(result_sd)
        return mean,sd,err

    # 同業種リストsimilarのすべての1ステップを推定する関数
    def onetimeallestimation(self,i):
        result = []
        result_sd = []
        err = 0
        for terget in self.similar:
            result.append(self.onetimeestimation(i,terget)[0])
            result_sd.append(self.onetimeestimation(i,terget)[1])
            err += self.onetimeestimation(i,terget)[2]
        return result,result_sd,err
    
## 線形回帰による方法
class Linearestimation:
    def __init__(self, df, similar, n):
        self.df=df
        self.similar = similar
        self.n = n
        
    def linearregression(self,train_x,train_y,test_x):
        clf = linear_model.LinearRegression()
        clf.fit(train_x,train_y)
        return np.dot(clf.coef_,test_x)

    def linearestimation(self,terget,n,i):
        train = self.similar[:]
        train.remove(terget)
        train_x = self.df[train].values[i-n:i]
        train_y = self.df[terget].values[i-n:i]
        test_x = self.df[train].values[i:i+1][0]
        result = self.linearregression(train_x,train_y,test_x)
        err = (result - self.df[terget].values[i+1])**2
        return result, err
    
    def linearonetimeallestimation(self,i):
        result = np.array([])
        err = 0
        for terget in similar:
            result = np.append(result,self.linearestimation(terget,self.n,i)[0])
            err += self.linearestimation(terget,self.n,i)[1]
        return result,err

## LASSOによる方法
class Lassoestimation:
    def __init__(self, df, similar, n):
        self.df = df
        self.similar = similar
        self.n = n        
        
    def lassoregression(self,train_x,train_y,test_x):
        lasso = linear_model.Lasso(alpha=0.00001)
        lasso.fit(train_x,train_y)
        return np.dot(lasso.coef_,test_x)

    def lassoestimation(self,terget,n,i):
        train = self.similar[:]
        train.remove(terget)
        train_x = self.df[train].values[i-n:i]
        train_y = self.df[terget].values[i-n:i]
        test_x = self.df[train].values[i:i+1][0]
        result = self.lassoregression(train_x,train_y,test_x)
        err = (result - self.df[terget].values[i+1])**2
        return result, err
    
    def lassoonetimeallestimation(self,i):
        result = np.array([])
        err = 0
        for terget in similar:
            result = np.append(result,self.lassoestimation(terget,self.n,i)[0])
            err += self.lassoestimation(terget,self.n,i)[1]
        print(err)
        return result,err

# GPRcheck
PortRet = []
GPR = GPRestimation(df,similar,10,3,16)
err = 0
for i in range(po_start,po_end+1):
    tmpdf = df[i:i+1]
    result = GPR.onetimeallestimation(i)[0]
    result_sd = GPR.onetimeallestimation(i)[1]
    err += GPR.onetimeallestimation(i)[2]
    long = similar[np.argmax(result)]
    short = similar[np.argmin(result)]
    portreturn = tmpdf[long].values[-1] - tmpdf[short].values[-1]
    PortRet.append(portreturn)
    print(str(i)+"...finished")
    print(err)
PortRet = pd.DataFrame(PortRet)
PortRet.index = index
PortRet.columns = ['Ret-GPR']  
PortRet.to_csv('Result-GPR.csv')
GPRerr = err

 # linearcheck
PortRet = []
Line = Linearestimation(df,similar,10)
err = 0
for i in range(po_start,po_end+1):
    tmpdf = df[i-11:i]
    result = Line.linearonetimeallestimation(i)[0]
    err += Line.linearonetimeallestimation(i)[1]
    long = similar[np.argmax(result)]
    short = similar[np.argmin(result)]
    portreturn = tmpdf[long].values[-1] - tmpdf[short].values[-1]
    PortRet.append(portreturn)
    print(str(i)+"...finished")
    print(err)
print(err)
PortRet = pd.DataFrame(PortRet)
PortRet.index = index
PortRet.columns = ['Ret-Linear']
PortRet.to_csv('Result-Linear.csv')
Linerr = err

# Lassocheck
PortRet = []
Lasso = Lassoestimation(df,similar,10)
err = 0
for i in range(po_start,po_end+1):
    tmpdf = df[i-11:i]
    result = Lasso.lassoonetimeallestimation(i)[0]
    err += Lasso.lassoonetimeallestimation(i)[1]
    long = similar[np.argmax(result)]
    short = similar[np.argmin(result)]
    portreturn = tmpdf[long].values[-1] - tmpdf[short].values[-1]
    PortRet.append(portreturn)
    print(str(i)+"...finished")
    print(err)
PortRet = pd.DataFrame(PortRet)
PortRet.index = index
PortRet.columns = ['Ret-Lasso'] 
PortRet.to_csv('Result-Lasso.csv')
Laserr = err
print(Linerr,Laserr,GPRerr)
