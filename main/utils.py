# -*- coding = utf-8 -*-

import numpy as np

def softmax(x):
    if x.ndim == 2:
        tmp = np.max(x,axis=1)
        #必ず一次元の配列になってしまう
        # tmpの形に合わせて計算を工夫している
        out = np.exp(x.T - tmp) / np.sum(np.exp(x.T - tmp),axis=0)
        return out.T
    tmp = np.max(x)
    return np.exp( x-tmp)/np.sum(np.exp(x-tmp))

def cross_entropy_error(Y,t):
    _y = Y
    _t = t
    if Y.ndim == 1:
        # バッチ処理を想定
        _t = t.reshape(1,t.size)
        _y = Y.reshape(1,Y.size)

    if _t.size == _y.size:
        _t = _t.argmax(axis=1)
             
    batch_size = _y.shape[0]
    return -np.sum(np.log(_y[np.arange(batch_size), _t] + 1e-7)) / batch_size

def mean_squared_error(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.shape[0])
        t = t.reshape(1,t.shape[0])

    batch_size = y.shape[0]
    out = 0.5*np.sum((y-t)**2)/batch_size
    return min(out, 1000)

class Affine():
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self,x):
        self.x = x.reshape(x.shape[0],-1)
        out = np.dot(x,self.W) + self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx

class ReLU():
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.Y = None
        self.t = None
    
    def forward(self,X,t):
        self.Y = softmax(X)
        self.t = t
        self.loss = cross_entropy_error(self.Y,self.t)
        return self.loss 

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        # データ一個あたりの誤差を算出する
        dx = (self.Y - self.t)/batch_size
        return dx


class IdentityWithLoss():
    def __init__(self):
        self.loss = None
        self.Y = None
        self.t = None

    def forward(self,X,t):
        self.Y = X
        self.t = t
        self.loss = mean_squared_error(self.Y,self.t)
        return self.loss 

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        # データ一個あたりの誤差を算出する
        dx = (self.Y - self.t)/batch_size
        return dx

class SGD():
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]

class Momentum():
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]            

class AdaGrid():
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr *grads[key]/ np.sqrt(self.h[key] + 1e-7)


if __name__ == "__main__":
    a = np.array([[1010,1000,900],[1010,1000,900]])
    #print(softmax(a))

    t = np.array([[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]])
    y = np.array([[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0],[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]])
    # print(cross_entropy_error(y,t))
    # t = np.array([0,0,1,0,0,0,0,0,0,0])
    # y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])

    print(mean_squared_error(y,t))