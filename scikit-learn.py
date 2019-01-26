# cikit-lerningの使い方についてのサンプルコード

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor

def true_func(x1,x2):
    y = x1**2+x2**2
    return y

np.random.seed(1)
x1 = np.random.normal(0, 1., 20)
x2 = np.random.normal(0,1.,20)
x_train = np.c_[x1,x2]
y_train = true_func(x1,x2) + np.random.normal(loc=0, scale=.1, size=1)
x1test = np.linspace(-3., 3., 200).reshape(-1, 1)
x2test = np.linspace(-3., 3., 200).reshape(-1, 1)
x_test = np.c_[x1test,x2test]

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

# X は (n_samples, n_features) の shape に変形する必要がある
# clf.fit(x_train.reshape(-1, 1), y_train)

def plot_result(x_test, mean, std):
    plt.plot(x_test[:, 0], mean, color="C0", label="predict mean")
    plt.fill_between(x_test[:, 0], mean + std, mean - std, color="C0", alpha=.3,label= "1 sigma confidence")
    plt.plot(x_train, y_train, "o",label= "training data")

# 予測は平均値と、オプションで 分散、共分散 を得ることが出来る
pred_mean, pred_std= GPR_fit(x_train,y_train,x_test)

print(pred_mean,pred_std)

# plot_result(x_test, pred_mean, pred_std)
# plt.title("Scikit-learn による予測")
# plt.legend()
# plt.savefig("sklern_predict.png", dpi=150)


