from cibp import *
import scipy.io as sio


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

if __name__ == "__main__":
    data = make_mnist_small('mnistSmall.mat', 50)
    print data

