from cibp import *
import scipy.io as sio
import pdb, traceback, sys
import matplotlib.pyplot as plt
import numpy.random as npr

def make_mnist_small(path, num_per_cat):
    data = sio.loadmat(path)
    train_x = data['trainData']
    train_y = data['trainLabels']
    n = train_x.shape[0]
    cat = dict()
    for c in range(10):
        cat[c] = 0
    res_x = []
    res_y = []
    for ni in range(n):
        y = train_y[ni]
        x = train_x[ni]
        c = list(y).index(1)
        if cat[c] < num_per_cat:
            res_y.append(y)
            res_x.append(x)
            cat[c] += 1
    return {'trainData': res_x, 'trainLabels': res_y}

def normalize_data(data):
    def trim(x):
        edge = 0.05
        if x >= 1-edge:
            x = 1-edge
        elif x <= edge:
            x = edge
        return x
    for d in data:
        d[:] = map(trim, d)

def toy_example_weights():
    try:
        net = CIBPnet(alpha=lambda l: 1, beta=lambda l: 1, nv=1)
        net.arch = [1,3]
        net.vs = [np.array([100.0]), np.array([100.0,100.0,100.0])]
        net.Ws = [object(), np.ones((1.0,3.0))]
        net.gammas = [np.array([0.0]), np.array([0.0,0.0,0.0])]
        net.Zs = [object(), np.ones((1.0,3.0))]
        hiddens = [CIBPhidden(net, np.array([0.5]))]
        hidden = hiddens[0]
        hidden.layers[1].u = np.ones(3)
        hidden.activate_all()
        for it in range(100):
            net.sample_weights(hiddens)
        print net.Ws
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == "__main__":
    try:
        # data = make_mnist_small('mnistSmall.mat', 50)
        data = sio.loadmat('mnistTiny.mat')
        train_data = data['trainData']
        nv = train_data.shape[1]
        net = CIBPnet(alpha=lambda l: 10, beta=lambda l: 1, nv=nv)
        normalize_data(train_data)
        ind = npr.choice(range(train_data.shape[0]), 32, replace=False)
        net.train(train_data, num_iter=10, valids=ind)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

    


