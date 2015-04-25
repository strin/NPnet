# CIBP for learning network structure.
import numpy as np
import math
import numpy.random as npr
from functools import partial
from mh import *
import pdb
import warnings
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

# warnings.filterwarnings('error')
plt.ion()

sigmoid1 = lambda z: 1/(1+math.exp(-float(z)))
inv_sigmoid = lambda x: np.log(x)-np.log(1-x)
grad_sigmoid1 = lambda x: sigmoid(x) * (1-sigmoid(x))
to_float = np.vectorize(float)

def assure(condition):
    assert(condition)

def tile(W, height, width, nh, nw):
    img = np.zeros((height * nh, width * nw))
    for i in range(nh):
        for j in range(nw):
            if i * nw + j >= W.shape[1]:
                break
            img[i*height:(i+1)*height, j*width:(j+1)*width] = W[:, j * nh + i].reshape(height, width)
    return img



class CIBPvislayer(object):
    def __init__(self, hidden, x):
        self.iu = inv_sigmoid(x)
        self.u = x
        self.y = self.iu
        self.hidden = hidden
        self.net = hidden.net
        self.v = self.net.vs[0]
        self.layer = 0
        self.size = len(x)
        self.mh = []

    def activate(self):
        self.y = self.hidden.activate(self.layer)

    def sample(self):
        self.activate()

    def proposal_lhood_y(self, y, node):
        v = self.v[node]
        temp = self.iu[node]
        res = -.5 * v * (temp - y)**2
        return res


class CIBPlayer(object):
    def __init__(self, hidden, layer, num_node):
        self.y = np.array([])
        self.u = np.array([])
        self.iu = np.array([]) # inv_sigmoid loses precision, maintain iu = inv_sigmoid(u).
        self.hidden = hidden
        self.net = hidden.net
        self.layer = layer
        self.size = 0
        self.mh = []
        self.resize(num_node)

    def get_y(self):
        # assert(self.y == self.hidden.activate(self.layer))
        return self.y 

    def resize(self, num_node):
        '''resize the layer to num_node
            if num_node > self.size, then fill in the new nodes with prior samples.
            otherwise, trim the layer to num_node.
        '''
        self.y.resize(num_node, refcheck=False)
        self.u.resize(num_node, refcheck=False)
        self.iu.resize(num_node, refcheck=False)
        v = self.net.vs[self.layer]
        layer = self.layer
        gamma = self.net.gammas[layer]
        if num_node > self.size:
            for i in range(self.size, num_node):
                self.y[i] = gamma[i]
                if self.layer < len(self.net.arch)-1:
                    W = self.net.Ws[layer+1]
                    Z = self.net.Zs[layer+1]
                    u = self.hidden.layers[layer+1].u
                    self.y[i] += np.dot(W[i, :] * Z[i, :], u)
                self.iu[i] = npr.randn() * np.sqrt(1.0/v[i]) + self.y[i]
                self.u[i] = sigmoid1(self.iu[i])
                self.mh.append(BasicMH(partial(self.propose, node=i),
                                     partial(self.lhood, node=i),
                                     lambda x, old_x: self.proposal_lhood(x, node=i)))
        else:
            self.mh = self.mh[:num_node]
        self.size = num_node

    def acc_rate(self):
        acc = []
        for mh in self.mh:
            acc.append(mh.acc_rate())
        return np.mean(acc)

    def propose(self, x, node): # propose in inv_sigmoid space.
        v = self.net.vs[self.layer][node]
        std = np.sqrt(1.0/v)
        # self.y = self.hidden.activate(self.layer)
        y = self.y[node]
        temp = npr.normal(y, std)
        return temp

    def proposal_lhood(self, x, node):
        try:
            v = self.net.vs[self.layer][node]
            self.y = self.hidden.activate(self.layer)
            y = self.y[node]
            res = -.5 * v * (x - y)**2 + .5 * np.log(v)
        except Warning as e:
            print e
        return res

    def proposal_lhood_y(self, y, node):
        v = self.net.vs[self.layer][node]
        temp = self.iu[node]
        res = -.5 * v * (temp - y)**2
        return res

    def lhood(self, x, node):
        assert(self.layer > 0)
        assert(self.u[node] == sigmoid1(self.iu[node]))
        y = self.hidden.activate(self.layer-1, self.u)
        WZ = self.net.Ws[self.layer][:, node] * self.net.Zs[self.layer][:, node]
        y -= WZ * self.u[node]
        y += WZ * sigmoid1(x)
        prev_iu = self.hidden.layers[self.layer-1].iu
        v = self.net.vs[self.layer-1]
        return sum(-.5 * v * (y - prev_iu)**2 + .5 * np.log(v))

    def sample(self):
        self.y = self.hidden.activate(self.layer)
        for i in range(self.size):
            self.iu[i] = self.mh[i].sample(self.iu[i])
            self.u[i] = sigmoid1(self.iu[i])

    def activate(self):
        self.y = self.hidden.activate(self.layer)


class CIBPhidden(object):
    def __init__(self, net, x):
        self.net = net
        self.x = x
        self.layers = [CIBPvislayer(self, x)]
        self.resize(net.arch)

    def resize(self, arch):
        layers = self.layers
        if len(arch) > len(layers):
            old_size = len(layers)
            layers.extend([None] * (len(arch) - len(layers)))
            for i in range(len(arch)-1, old_size-1, -1):
                layer = CIBPlayer(self, i, arch[i])
                layers[i] = layer
        self.layers = layers[:len(arch)]
        for i in range(len(arch)-1, 0, -1):
            self.layers[i].resize(arch[i])

    def activate(self, layer, next_u=[], baseline=None, delta=lambda WZ: 0):
        res = np.array(self.net.gammas[layer])
        if layer < len(self.layers)-1:
            if len(next_u) == 0:
                next_u = self.layers[layer+1].u
            WZ = self.net.Ws[layer+1] * self.net.Zs[layer+1]
            res += np.dot(WZ, next_u)
        return res

    def activate_all(self):
        for (li, layer) in enumerate(self.layers):
            layer.activate()

    def sample(self):
        for (li, layer) in enumerate(self.layers):
            layer.sample()
        self.activate_all()
    
    def acc_rate(self):
        acc = []
        for layer in self.layers[1:]:
            acc.append(layer.acc_rate())
        return np.mean(acc)

    def lhood(self):
        arch = self.net.arch
        lhood = 0
        n = len(arch)
        for i in range(arch[n-1]):
            iu = self.layers[n-1].iu
            lhood += self.layers[n-1].proposal_lhood(iu[i], i)
        for l in range(1, len(arch)):
            iu = self.layers[l].iu
            for i in range(arch[l]):
                lhood += self.layers[l].lhood(iu[i], i)
        return lhood

class CIBPnet(object):
    def __init__(self, **param):
        assert(isinstance(param, dict))
        self.a = param["a"] if "a" in param else lambda layer: 1
        self.b = param["b"] if "b" in param else lambda layer: 1
        self.alpha = param["alpha"] if "alpha" in param else lambda layer: 1
        self.beta = param["beta"] if "beta" in param else lambda layer: 5
        self.mu_w = param["mu_w"] if "mu_w" in param else lambda layer: 0
        self.rho_w = param["rho_w"] if "rho_w" in param else lambda layer: 1
        self.mu_gamma = param["mu_gamma"] if "mu_gamma" in param else lambda layer: 0
        self.rho_gamma = param["rho_gamma"] if "rho_gamma" in param else lambda layer: 1
        self.num_visible = param["nv"]

        self.vs = [np.array([])]
        self.Ws = [object()]
        self.gammas = [np.array([])]
        self.Zs = [object()]

        self.arch = [0]
        for i in range(self.num_visible):
            self.sample_prior_cibp(0)

    def sample_prior_cibp(self, layer):
        "add a new node at layer k, and start recursive sampling from CIBP prior."
        arch = self.arch
        self.gammas[layer].resize(arch[layer]+1, refcheck=False)
        self.vs[layer].resize(arch[layer]+1, refcheck=False)
        if layer > 0:
            K = arch[layer]
            Kp = arch[layer-1]
            self.Ws[layer].resize(Kp, K+1, refcheck=False)
            self.Ws[layer][:, K] = self.sample_prior_W(1, Kp, layer)
            self.Zs[layer].resize(Kp, K+1, refcheck=False)
            self.Zs[layer][:, K] = np.zeros(Kp)
        self.gammas[layer][-1:] = self.sample_prior_gamma(1, layer)
        self.vs[layer][-1:] = self.sample_prior_v(1, layer)
        new_dish = 1
        layer += 1
        while new_dish > 0:
            arch[layer-1] += new_dish
            if layer >= len(self.arch):
                self.arch.append(0)
                self.Ws.append(np.array([]))
                self.Zs.append(np.array([]))
                self.gammas.append(np.array([]))
                self.vs.append(np.array([]))
            W = self.Ws[layer]
            gamma = self.gammas[layer]
            v = self.vs[layer]
            Z = self.Zs[layer]
            alpha = self.alpha(layer)
            beta = self.beta(layer)
            K = arch[layer-1]
            Kp = arch[layer]
            if Kp > 0:
                sum_Z = Z.sum(axis=0)
                W.resize(K, Kp, refcheck=False)
                W[K-1, :] = self.sample_prior_W(1, Kp, layer)
                Z.resize(K, Kp, refcheck=False)
                for kp in range(Kp):
                    prob = sum_Z[kp] / (K-1 + beta)
                    if npr.rand() < prob:
                        Z[K-1, kp] = 1
            new_dish = npr.poisson(alpha * beta / (K-1 + beta))
            W.resize(K, Kp + new_dish, refcheck=False)
            W[K-1, Kp:Kp+new_dish] = self.sample_prior_W(1, new_dish, layer)
            Z.resize(K, Kp + new_dish, refcheck=False)
            Z[K-1, Kp:Kp+new_dish] = 1
            gamma.resize(Kp+new_dish, refcheck=False)
            gamma[Kp:Kp+new_dish] = self.sample_prior_gamma(new_dish, layer)
            v.resize(Kp+new_dish, refcheck=False)
            v[Kp:Kp+new_dish] = self.sample_prior_v(new_dish, layer)
            layer += 1
        if arch[-1] == 0:
            arch.pop()

    def resize(self, arch):
        self.arch[:] = arch
        for l in range(len(arch)):
            if l >= 1:
                self.Ws[l].resize((arch[l-1], arch[l]), refcheck=False)
                self.Zs[l].resize((arch[l-1], arch[l]), refcheck=False)
            self.gammas[l].resize(arch[l], refcheck=False)
            self.vs[l].resize(arch[l], refcheck=False)

    def sample_prior_gamma(self, K, layer):
        return npr.randn(K) * np.sqrt(1.0 / self.rho_gamma(layer)) + self.mu_gamma(layer)

    def sample_prior_W(self, K, Kp, layer):
        return npr.randn(K, Kp) * np.sqrt(1.0 / self.rho_w(layer)) + self.mu_w(layer)

    def sample_prior_v(self, K, layer):
        return npr.gamma(self.a(layer), 1.0/self.b(layer), K)

    def sample_hiddens(sef, hiddens):
        lhood = []
        for hidden in hiddens:
            hidden.sample()
            lhood.append(hidden.lhood())
        return np.mean(lhood)

    def activate_hidden(self, hiddens):
        for hidden in hiddens:
            hidden.activate_all()

    def acc_rate_hiddens(self, hiddens):
        acc = []
        for hidden in hiddens:
            acc.append(hidden.acc_rate())
        return np.mean(acc)

    def sample_weights(self, hiddens):
        arch = self.arch
        for l in range(1, len(arch)):
            W = self.Ws[l]
            Z = self.Zs[l]
            WZ = W * Z
            mu_w = self.mu_w(l)
            rho_w = self.rho_w(l)
            for kp in range(arch[l]):
                A = np.zeros_like(self.gammas[l-1], dtype='float')
                B = np.zeros_like(self.gammas[l-1], dtype='float')
                for hidden in hiddens:
                    prev_iu = hidden.layers[l-1].iu
                    hidden.layers[l-1].activate()
                    prev_y = hidden.layers[l-1].y
                    u = hidden.layers[l].u
                    A += u[kp] * (prev_iu - prev_y + WZ[:, kp] * u[kp])
                    B += u[kp]**2
                A *= self.vs[l-1]
                B *= self.vs[l-1]
                rho_w_post = rho_w + B
                std_w_post = np.sqrt(1.0/rho_w_post)
                mu_w_post = (rho_w * mu_w + A) / rho_w_post
                W[:, kp] = npr.randn(arch[l-1]) * std_w_post + mu_w_post
        self.activate_hidden(hiddens)

    def sample_biases(self, hiddens):
        N = len(hiddens)
        arch = self.arch
        for l in range(len(arch)):
            W = self.Ws[l]
            gamma = self.gammas[l]
            v = self.vs[l]
            mu_gamma = self.mu_gamma(l)
            rho_gamma = self.rho_gamma(l)
            A = 0
            for hidden in hiddens:
                iu = hidden.layers[l].iu
                y = hidden.layers[l].y
                A += iu - y + gamma
            A *= v
            rho_gamma_post = rho_gamma + N * v
            std_b_post = np.sqrt(1.0/rho_gamma_post)
            mu_gamma_post = (rho_gamma * mu_gamma + A) / rho_gamma_post
            gamma[:] = npr.randn(arch[l]) * std_b_post + mu_gamma_post
        self.activate_hidden(hiddens)

    def sample_activation_variance(self, hiddens):
        N = len(hiddens)
        arch = self.arch
        for l in range(len(arch)):
            v = self.vs[l]
            A = np.zeros(arch[l])
            for hidden in hiddens:
                iu = hidden.layers[l].iu
                y = hidden.layers[l].y
                A += (iu - y)**2
            a_post = np.ones(arch[l]) * self.a(l) + .5 * N
            b_post = self.b(l) + A * .5
            v[:] = np.array(map(npr.gamma, a_post, 1.0/b_post))

    def sample_structure(self, hiddens):
        arch = self.arch
        for l in range(len(self.arch)-1):
            ln = l+1
            Z = self.Zs[ln]
            W = self.Ws[ln]
            WZ = W * Z
            # # compute diff = iu - y
            # diff = np.zeros_like(iu)
            # for hidden in hiddens:
            #     diff += hidden.layers[l].iu - hidden.layers[l].get_y()

            for k in range(arch[l]):
                singleton = []
                for kp in range(arch[ln]):
                    if sum(Z[:, kp]) == 0:  # ghost parent.
                        continue
                    eta = sum(Z[:, kp]) - Z[k, kp]
                    if eta == 0:    # singleton parent.
                        singleton.append(kp)
                    else:
                        # Phase I. Adding edges to non-singletons.
                        M = arch[l] + self.beta(l) - 1
                        lhood0 = np.log(eta) - np.log(M)
                        lhood1 = np.log(M-eta) - np.log(M)
                        for hidden in hiddens:
                            next_u = hidden.layers[ln].u
                            y = hidden.layers[l].y
                            baseline0 = y[k]
                            baseline0 -= WZ[k, kp] * next_u[kp]     # Z = 0.
                            baseline1 = baseline0 + W[k, kp] * next_u[kp]    # Z = 1.
                            lhood0 += hidden.layers[l].proposal_lhood_y(baseline0, k)
                            lhood1 += hidden.layers[l].proposal_lhood_y(baseline1, k)
                        max_lhood = max(lhood0, lhood1)
                        lhood0 = lhood0 - max_lhood
                        lhood1 = lhood1 - max_lhood
                        prob = np.exp(lhood1) / (np.exp(lhood0) + np.exp(lhood1))
                        if npr.rand() < prob:
                            Z[k, kp] = 1
                        else:
                            Z[k, kp] = 0

                continue
                # Phase II. remove or add singleton parents.
                Ko = len(singleton)
                if npr.rand() < .5:     # birth.
                    old_arch = list(self.arch)
                    j = old_arch[ln]
                    self.sample_prior_cibp(ln)
                    self.check_shapes()
                    Z = self.Zs[ln]
                    Z[k, j] = 1
                    W = self.Ws[ln]
                    WZ = W * Z
                    lhood = np.log(self.alpha(l) * self.beta(l)) - \
                        2 * np.log(Ko+1) - np.log(self.beta(l) + self.arch[l] - 1)
                    for hidden in hiddens:
                        hidden.resize(arch)
                        assert(hidden.layers[1].y.shape[0] == arch[1])
                        next_u = hidden.layers[ln].u
                        y = hidden.layers[l].y
                        baseline0 = y[k]
                        bsaeline1 = y[k] + W[k, j] * next_u[j]
                        lhood -= hidden.layers[l].proposal_lhood_y(baseline0, k)
                        lhood += hidden.layers[l].proposal_lhood_y(baseline1, k)
                    if npr.rand() >= np.exp(lhood):  # reject, roll back changes.
                        self.resize(old_arch)
                        for hidden in hiddens:
                            hidden.resize(old_arch)
                        self.check_shapes()
                elif Ko > 0:   # death.
                    ind = npr.choice(Ko)
                    lhood = 2 * np.log(Ko) + np.log(self.beta(l) + self.arch[l] - 1) \
                        - np.log(self.alpha(l)) - np.log(self.beta(l))
                    for hidden in hiddens:
                        next_u = hidden.layers[ln].u
                        y = hidden.layers[l].y
                        baseline0 = y[k] - W[k, ind] * Z[k, ind] * next_u[ind]
                        baseline1 = y[k]
                        lhood += hidden.layers[l].proposal_lhood_y(baseline0, k)
                        lhood -= hidden.layers[l].proposal_lhood_y(baseline1, k)
                    if npr.rand() < np.exp(lhood):  # accept, remove edge.
                        Z[k, ind] = 0

    def check_shapes(self):
        for l in range(len(self.arch)):
            if l >= 1:
                assert(self.Ws[l].shape[0] == self.arch[l-1])
                assert(self.Ws[l].shape[1] == self.arch[l])
                assert(self.Zs[l].shape[0] == self.arch[l-1])
                assert(self.Zs[l].shape[1] == self.arch[l])
            assert(self.gammas[l].shape[0] == self.arch[l])
            assert(self.vs[l].shape[0] == self.arch[l])

    def train(self, examples, num_iter):
        hiddens = []
        print 'arch = ', self.arch
        for ex in examples:
            hidden = CIBPhidden(self, ex)
            hiddens.append(hidden)
        for it in range(num_iter):
            print >> stdout, 'iter = ', it
            stdout.flush()
            lhood = self.sample_hiddens(hiddens)
            print >> stdout, '... lhood = ', lhood
            print >> stdout, '... accept = ', self.acc_rate_hiddens(hiddens)
            self.sample_weights(hiddens)
            self.sample_biases(hiddens)
            self.sample_activation_variance(hiddens)
            # self.sample_structure(hiddens)
        plt.imshow(tile(self.Ws[1], 28, 28, 10, 10), cmap=cm.Greys_r)
        plt.show()
            
