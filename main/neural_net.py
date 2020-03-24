# -*- coding = utf-8 -*-

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from utils import Affine, ReLU, SoftmaxWithLoss, IdentityWithLoss, Momentum, AdaGrid
from import_data import get_test_data, get_train_data
# from text_modules.dataset.mnist import load_mnist

class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,weight_init=0.01):
        self.params = {}
        # self.params["W1"] = weight_init*np.random.rand(input_size,hidden_size)
        self.params["W1"] = np.random.randn(input_size, hidden_size)/ np.sqrt(input_size) * np.sqrt(2.0)
        self.params["b1"] = np.zeros(hidden_size)
        # self.params["W2"] = weight_init*np.random.rand(hidden_size,output_size)
        self.params["W2"] = np.random.randn(hidden_size, output_size)/ np.sqrt(hidden_size) * np.sqrt(2.0)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"],self.params["b1"])
        self.layers["ReLU"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"],self.params["b2"])

        self.last_layer = IdentityWithLoss()

    def predict(self,x):
        out = x.copy()
        for layer in self.layers.values():
            out = layer.forward(out)
        return out
    
    def loss(self,x,t):
        y = self.predict(x)
        out = self.last_layer.forward(y,t)
        return out
    
    def gradient(self,x,t):
        loss = self.loss(x,t)    
        dout = self.last_layer.backward(dout =1)

        # この方法でないと逆順にできない。OrderedDictのせい？
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        return grads

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1: t = np.argmax(t,axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy


if __name__ == "__main__":
    #
    x_train, t_train = get_train_data(1000)
    x_test, t_test = get_test_data()
    # (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=True)

    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 1000

    train_loss_rate = []
    train_accuracy = []
    test_accuracy = []
    epoc_rate = max(train_size/batch_size,1)

    network = TwoLayerNet(input_size=2,hidden_size=10,output_size=1)

    # Updator = Momentum(lr=0.15)
    Updator = AdaGrid(lr=0.15)

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch,t_batch)

        Updator.update(network.params, grads)
        # for key in grads.keys():
        #     network.params[key] -= learning_rate*grads[key]
        
        loss = network.loss(x_batch,t_batch)
        train_loss_rate.append(loss)

        if i%epoc_rate == 0:
            test_loss = network.loss(x_test, t_test)
            test_accuracy.append(test_loss)
            print("Test accuracy: %f"%test_loss)
            # accuracy = network.accuracy(x_test,t_test)
            # test_accuracy.append(accuracy)
            # print(accuracy)
            # print("Test accuracy: %f"%accuracy)

    _x = range(len(train_loss_rate))
    plt.plot(_x,train_loss_rate)
    plt.ylim(0,100)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Number of Iterations")
    plt.show()

    test_show = [[10.0,22.0],[3.0,1.0],[11.0,100.0]]
    for ary in test_show:
        ans = network.predict(np.array(ary))
        print("##############################")
        print("Mass: %f, Stiffness: %f"%(ary[0],ary[1]))
        print("Predicted :%s"%str(ans[0]))
        print("Answer :%s"%str(2.0*np.pi*(ary[0]/ary[1])**0.5))
