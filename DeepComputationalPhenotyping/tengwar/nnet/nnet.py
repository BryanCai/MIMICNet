# A package including neural network classes, which provide the same methods as what sklearn does

import numpy as np

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def threshold(Z):
    return (Z>=0.5).astype(int)

def softmax(Z):
    eZ = np.exp(Z)
    eZ = eZ / np.sum(eZ,axis=1)[:,None]
    return eZ

def argmax(Z):
    return Z.argmax(axis=1)

# simple feed-forward network, used to get network outputs and evaluations
class FeedForwardNetwork(object):
    def __init__(self, W_h=[], b_h=[], W_o=None, b_o=None, fn_h=sigmoid, fn_o=sigmoid, df=threshold):
        # weights and transform functions for hidden layer(s)
        self.W_h = np.array(W_h) if type(W_h) is not np.ndarray else W_h
        self.b_h = np.array(b_h) if type(b_h) is not np.ndarray else b_h
        assert(W_h.shape[0] == b_h.shape[0])
        self.fn_h = fn_h
        
        # weights and transform function for output layer
        self.W_o = np.array(W_o) if type(W_o) is not np.ndarray else W_o
        self.b_o = np.array(b_o) if type(b_o) is not np.ndarray else b_o
        if self.W_o is None or self.b_o is None:
            self.fn_o = None
            self.df = None
        else:
            self.fn_o = fn_o
            self.df = df

    # get the i-th (hidden) layer output as the features of input X
    def transform_features(self, X, layers=None):
        A = X
        if layers is None or layers < 1 or layers > self.W_h.shape[0]:
            layers = self.W_h.shape[0]

        l = 1
        for W,b in zip(self.W_h, self.b_h):
            if l > layers:
                break
            s = A.shape
            A = self.fn_h(np.dot(A, W) + b)
            #print l, s, A.shape
            l += 1
        return A

    def fit(self, X, y):
        pass
        
    # get output layer values of input X
    def decision_function(self, X):
        A = self.transform_features(X)
        if self.fn_o is not None:
            return self.fn_o(np.dot(A, self.W_o) + self.b_o)
        return np.zeros((X.shape[0],))


    # get prediction of input X (compare output value and threshold)
    def predict(self, X):
        dv = self.decision_function(X)
        return self.df(dv)        
        
    # a static constructor of feed forward network given weights and functions
    @staticmethod
    def from_saved_weights(fn, fn_h=sigmoid, fn_o=sigmoid, df=threshold):
        ndata = np.load(fn)
        W_h = ndata['W_h'] if 'W_h' in ndata else (ndata['Wh'] if 'Wh' in ndata else ndata['W'])
        b_h = ndata['b_h'] if 'b_h' in ndata else (ndata['bh'] if 'bh' in ndata else ndata['b'])
        W_o = ndata['W_o'] if 'W_o' in ndata else (ndata['Wo'] if 'Wo' in ndata else None)
        b_o = ndata['b_o'] if 'b_o' in ndata else (ndata['bo'] if 'bo' in ndata else None)
        return FeedForwardNetwork(W_h=W_h, b_h=b_h, W_o=W_o, b_o=b_o, fn_h=fn_h, fn_o=fn_o, df=df)
