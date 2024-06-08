import multiprocessing
import numpy as np
from tqdm import tqdm
import pickle
import random

from CNN.PoolLayer2D import *
from CNN.Convolution2D import *
from CNN.ActivationLayer import *

class CNN:
    """
    Convulution islemi tamamlandiktan sonra
    Fully Connected Layer adimini bu class' ta gerceklestiri
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
               kernel_size=(5, 5), pool_size: tuple = (2, 2),
               img_depth: int = 1, num_filt2: int = 5, num_filt1: int = 5,
               alpha: float = 0.0001):

        
        self.X = X
        self.y = y
        self.train_data = np.hstack((X, y))
        self.X_val = X_val
        self.y_val = y_val

        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.img_depth = img_depth
        self.num_filt2 = num_filt2
        self.num_filt1 = num_filt1
        self.alpha = alpha

        self.init_params()

    def init_params(self):
        self.conv2D = Convolution2D()
        self.pool2D = PoolLayer2D()

        self.f1, self.f2, self.w3, self.w4 = (self.num_filt1 ,self.img_depth,*self.kernel_size), (self.num_filt2 ,self.num_filt1,*self.kernel_size), (10, 500), (10, 10)
        self.f1 = self.initializeFilter(self.f1)
        self.f2 = self.initializeFilter(self.f2)
        self.w3 = self.initializeWeight(self.w3)
        self.w4 = self.initializeWeight(self.w4)

        self.b1 = np.zeros((self.f1.shape[0],1))
        self.b2 = np.zeros((self.f2.shape[0],1))
        self.b3 = np.zeros((self.w3.shape[0],1))
        self.b4 = np.zeros((self.w4.shape[0],1))
        
        self.t = 0
        
        self.v1 = np.zeros(self.f1.shape)
        self.v2 = np.zeros(self.f2.shape)
        self.v3 = np.zeros(self.w3.shape)
        self.v4 = np.zeros(self.w4.shape)
        self.bv1 = np.zeros(self.b1.shape)
        self.bv2 = np.zeros(self.b2.shape)
        self.bv3 = np.zeros(self.b3.shape)
        self.bv4 = np.zeros(self.b4.shape)
        
        self.s1 = np.zeros(self.f1.shape)
        self.s2 = np.zeros(self.f2.shape)
        self.s3 = np.zeros(self.w3.shape)
        self.s4 = np.zeros(self.w4.shape)
        self.bs1 = np.zeros(self.b1.shape)
        self.bs2 = np.zeros(self.b2.shape)
        self.bs3 = np.zeros(self.b3.shape)
        self.bs4 = np.zeros(self.b4.shape)

    def initializeFilter(self, size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def initializeWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def forward_prop(self, image, conv_stride, pool_size, pool_stride):
        conv1 = self.conv2D.convolution(image, self.f1, self.b1, conv_stride)
        conv1 = ActivationLayer.ReLU(conv1)

        conv2 = self.conv2D.convolution(conv1, self.f2, self.b2, conv_stride)
        conv2 = ActivationLayer.ReLU(conv2)

        pooled = self.pool2D.maxpool(conv2, pool_size, pool_stride) 

        (nf2, dim1, dim2) = pooled.shape
        fc = pooled.reshape((nf2 * dim1 * dim2, 1)) 
        
        z = self.w3.dot(fc) + self.b3
        z = ActivationLayer.ReLU(z)

        out = self.w4.dot(z) + self.b4
        probs = ActivationLayer.softmax(out)
        
        return (conv1, conv2, pooled, fc, z), probs

    def categorical_cross_entropy(self, probs, label, epsilon=1e-10):
        return -np.sum(label * np.log(probs))
    

    def backward_prop(self, params, image, label, conv_stride, pool_size, pool_stride):
        """
            Egime gore degismesi gereken araligi bul 
        """
        conv1, conv2, pooled, fc, z, probs = params
        
        dout = probs - label
        dw4 = dout.dot(z.T)
        db4 = np.sum(dout, axis=1).reshape(self.b4.shape)

        dz = self.w4.T.dot(dout)
        dz[z<=0] = 0

        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis=1).reshape(self.b3.shape)

        dFc = self.w3.T.dot(dz)
        dpool = dFc.reshape(pooled.shape)

        dconv2 = self.pool2D.maxpoolBackward(dpool, conv2, pool_size, pool_stride) 
        dconv2[conv2<=0] = 0

        dconv1, df2, db2 = self.conv2D.convolutionBackward(dconv2, conv1, self.f2, conv_stride)
        dconv1[conv1<=0] = 0

        dimage, df1, db1 = self.conv2D.convolutionBackward(dconv1, image, self.f1, conv_stride)

        return df1, db1, df2, db2, dw3, db3, dw4, db4
    
    
    def init_ADAM_params(self):
        df1 = np.zeros(self.f1.shape)
        df2 = np.zeros(self.f2.shape)
        dw3 = np.zeros(self.w3.shape)
        dw4 = np.zeros(self.w4.shape)
        db1 = np.zeros(self.b1.shape)
        db2 = np.zeros(self.b2.shape)
        db3 = np.zeros(self.b3.shape)
        db4 = np.zeros(self.b4.shape)
        
        return df1, df2, dw3, dw4, db1, db2, db3, db4

    def update_ADAM_parameters(self, beta1, beta2, batch_size, df1, df2, dw3, dw4, db1, db2, db3, db4):
        self.t += 1
        
        self.v1 = beta1*self.v1 + (1-beta1)*df1/batch_size
        self.s1 = beta2*self.s1 + (1-beta2)*(df1/batch_size)**2
        v1_hat = self.v1 / (1 - beta1 ** self.t)
        s1_hat = self.s1 / (1 - beta2 ** self.t)
        self.f1 -= self.alpha * v1_hat/np.sqrt(s1_hat+1e-7)
        
        self.bv1 = beta1*self.bv1 + (1-beta1)*db1/batch_size
        self.bs1 = beta2*self.bs1 + (1-beta2)*(db1/batch_size)**2
        bv1_hat = self.bv1 / (1 - beta1 ** self.t)
        bs1_hat = self.bs1 / (1 - beta2 ** self.t)
        self.b1 -= self.alpha * bv1_hat/np.sqrt(bs1_hat+1e-7)
    
        self.v2 = beta1*self.v2 + (1-beta1)*df2/batch_size
        self.s2 = beta2*self.s2 + (1-beta2)*(df2/batch_size)**2
        v2_hat = self.v2 / (1 - beta1 ** self.t)
        s2_hat = self.s2 / (1 - beta2 ** self.t)
        self.f2 -= self.alpha * v2_hat/np.sqrt(s2_hat+1e-7)
                        
        self.bv2 = beta1*self.bv2 + (1-beta1) * db2/batch_size
        self.bs2 = beta2*self.bs2 + (1-beta2)*(db2/batch_size)**2
        bv2_hat = self.bv2 / (1 - beta1 ** self.t)
        bs2_hat = self.bs2 / (1 - beta2 ** self.t)
        self.b2 -= self.alpha * bv2_hat/np.sqrt(bs2_hat+1e-7)
        
        self.v3 = beta1*self.v3 + (1-beta1) * dw3/batch_size
        self.s3 = beta2*self.s3 + (1-beta2)*(dw3/batch_size)**2
        v3_hat = self.v3 / (1 - beta1 ** self.t)
        s3_hat = self.s3 / (1 - beta2 ** self.t)
        self.w3 -= self.alpha * v3_hat/np.sqrt(s3_hat+1e-7)
        
        self.bv3 = beta1*self.bv3 + (1-beta1) * db3/batch_size
        self.bs3 = beta2*self.bs3 + (1-beta2)*(db3/batch_size)**2
        bv3_hat = self.bv3 / (1 - beta1 ** self.t)
        bs3_hat = self.bs3 / (1 - beta2 ** self.t)
        self.b3 -= self.alpha * bv3_hat/np.sqrt(bs3_hat+1e-7)
        
        self.v4 = beta1*self.v4 + (1-beta1) * dw4/batch_size
        self.s4 = beta2*self.s4 + (1-beta2)*(dw4/batch_size)**2
        v4_hat = self.v4 / (1 - beta1 ** self.t)
        s4_hat = self.s4 / (1 - beta2 ** self.t)
        self.w4 -= self.alpha * v4_hat / np.sqrt(s4_hat+1e-7)
        
        self.bv4 = beta1*self.bv4 + (1-beta1)*db4/batch_size
        self.bs4 = beta2*self.bs4 + (1-beta2)*(db4/batch_size)**2
        bv4_hat = self.bv4 / (1 - beta1 ** self.t)
        bs4_hat = self.bs4 / (1 - beta2 ** self.t)
        self.b4 -= self.alpha * bv4_hat / np.sqrt(bs4_hat+1e-7)
 
    def adam_optimizer_multi(self, chunk_range, conv_stride, pool_size, pool_stride, X, Y):
        cost = 0

        df1 = np.zeros(self.f1.shape)
        df2 = np.zeros(self.f2.shape)
        dw3 = np.zeros(self.w3.shape)
        dw4 = np.zeros(self.w4.shape)
        db1 = np.zeros(self.b1.shape)
        db2 = np.zeros(self.b2.shape)
        db3 = np.zeros(self.b3.shape)
        db4 = np.zeros(self.b4.shape)

        for i in range(*chunk_range):
            x = X[i]
            y = np.eye(10)[int(Y[i])].reshape(10, 1)

            params, probs = self.forward_prop(x, conv_stride, pool_size, pool_stride)
            loss = self.categorical_cross_entropy(probs, y)
            df1_, db1_, df2_, db2_, dw3_, db3_, dw4_, db4_ = self.backward_prop((*params, probs), x, y, conv_stride, pool_size, pool_stride)

            df1+=df1_
            db1+=db1_
            df2+=df2_
            db2+=db2_
            dw3+=dw3_
            db3+=db3_
            dw4+=dw4_
            db4+=db4_

            cost += loss

        return chunk_range, df1, db1, df2, db2, dw3, db3, dw4, db4, cost 

    def adam_optimizer(self, batch, costs, conv_stride, pool_size, pool_stride, t, epoch, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8):
        X = batch[:,0:-1] 
        X = X.reshape(len(batch), 1, 28, 28)
        Y = batch[:,-1]

        batch_size = len(batch)


        num_processes = multiprocessing.cpu_count()
        processes = []
        chunk_size = batch_size // num_processes

        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                start = i * chunk_size
                end = min(start + chunk_size, batch_size)

                processes.append(pool.apply_async(self.adam_optimizer_multi, ((start, end), conv_stride, pool_size, pool_stride, X, Y)))

            for i, process in enumerate(processes):
                df1, df2, dw3, dw4, db1, db2, db3, db4 = self.init_ADAM_params()
                (start, end), df1, db1, df2, db2, dw3, db3, dw4, db4, cost = process.get()

                cost = cost / (end- start)
                self.update_ADAM_parameters(beta1, beta2, (end- start), df1, df2, dw3, dw4, db1, db2, db3, db4)
                
                t.set_description("Epoch: %d/%d, Cost: %.4f, Batch Size: %d/%d, Alpha: %f " % ((epoch +1), epochs, cost, (i+1)*(end- start), batch_size, self.alpha))
        
        pool.close()
        pool.join()
        
        costs.append(cost)
        
        return costs


    def accuracy(self, conv_stride, pool_size, pool_stride, val_costs):
        val_cost = 0
        t = tqdm(range(len(self.X_val)), leave=True)
        for i in t:
            x = self.X_val[i]
            y = np.eye(10)[int(self.y_val[i])].reshape(10, 1)
            _, probs = self.forward_prop(x, conv_stride, pool_size, pool_stride)
            val_cost += self.categorical_cross_entropy(probs, y)
            t.set_description(f"Computing validation cost {val_cost/(i+1):.4f}")
        val_cost /= len(self.X_val)
        val_costs.append(val_cost)
        
        return val_cost

    def train(self, epochs=10, batch_size=1024, conv_stride=1, pool_size=2, pool_stride=2, beta1=0.9, beta2=0.999, patience=3):
        val_costs, costs = [], []
        best_val_cost = float('inf')
        epochs_no_improve = 0

        print("LR:"+str(self.alpha)+", Batch Size:"+str(batch_size))
        
        for epoch in range(epochs):
            np.random.shuffle(self.train_data)
            batches = [self.train_data[k:k + batch_size] for k in range(0, self.X.shape[0], batch_size)]
            
            t = tqdm(batches, leave=True)
            for index, batch in enumerate(t):
                self.adam_optimizer(batch, costs, conv_stride, pool_size, pool_stride, t, epoch, epochs, beta1, beta2)
                t.set_description("Epoch: %d/%d, Cost: %.4f, Batch Size: %d/%d, Alpha: %f" % ((epoch +1), epochs, costs[-1], batch_size, batch_size, self.alpha))
            
            val_cost = self.accuracy(conv_stride, pool_size, pool_stride, val_costs)
            
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return costs
    
    def predict(self, image, conv_s = 1, pool_f = 2, pool_s = 2):
        _, probs = self.forward_prop(image, conv_s, pool_f, pool_s)
        return np.argmax(probs), np.max(probs), probs
