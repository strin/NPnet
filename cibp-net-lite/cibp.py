# CIBP for learning network structure.
import numpy as np
import numpy.random as npr
from functools import partial
from mh import *
import pdb

sigmoid = lambda z: 1/(1+np.exp(-float(z)))
inv_sigmoid = lambda x: np.log(x)-np.log(1-x)
grad_sigmoid = lambda x: sigmoid(x) * (1-sigmoid(x))
to_float = np.vectorize(float)

class CIBPlayer(object):
    def __init__(self, hidden, layer, num_node):
        self.y = np.array([])
        self.u = np.array([])
        self.hidden = hidden
        self.net = hidden.net
        self.v = self.net.vs[layer]
        self.layer = layer
        self.size = 0
        self.mh = []
        self.resize(num_node)

    def resize(self, num_node):
        '''resize the layer to num_node
            if num_node > self.size, then fill in the new nodes with prior samples.
            otherwise, trim the layer to num_node.
        '''
        self.y.resize(num_node)
        self.u.resize(num_node)
        layer = self.layer
        gamma = self.net.gammas[layer]
        if num_node > self.size:
            for i in range(self.size, num_node):
                self.y[i] = gamma
                if self.layer < len(self.net.arch)-1:
                    W = self.net.Ws[layer+1]
                    Z = self.net.Zs[layer+1]
                    u = self.hidden.layers[layer+1].u
                    self.y[i] += W[i, :] * Z[i, :] * u
                self.u[i] = npr.randn() * np.sqrt(1.0/self.v[i]) + self.y[i]
                self.mh[i] = BasicMH(partial(self.propose, node=i),
                                     partial(self.lhood, node=i),
                                     partial(self.proposal_lhood, node=i))
        self.size = num_node

    def propose(self, x, node):
        v = self.v[node]
        std = np.sqrt(1.0/v)
        self.y = self.hidden.activate(self.layer)
        y = self.y[node]
        temp = npr.normal(y, std)
        return sigmoid(temp)

    def proposal_lhood(self, x, node):
        v = self.v[node]
        self.y = self.hidden.activate(self.layer)
        y = self.y[node]
        temp = inv_sigmoid(x)
        res = -.5 * v * (temp - y)**2 - np.log(grad_sigmoid(temp))
        return res

    def proposal_lhood_y(self, y, node):
        v = self.v[node]
        x = self.u[node]
        temp = inv_sigmoid(x)
        res = -.5 * v * (temp - y)**2 - np.log(grad_sigmoid(temp))
        return res

    def lhood(self, x, node):
        assert(self.layer > 0)
        u = self.u
        u[node] = x
        y = self.hidden.activate(self.layer-1, u)
        prev_u = self.net.layers[self.layer-1].u
        v = self.net.layers[self.layer-1].v
        return sum(-.5 * v * (y - prev_u)**2)

    def sample(self):
        self.y = self.hidden.activate()
        for i in range(self.size):
            self.u[i] = self.mh[i].sample(self.u[i])


class CIBPhidden(object):
    def __init__(self, net, x):
        self.net = net
        self.x = x
        self.layers = [x]
        self.resize(net.arch)

    def resize(self, arch):
        layers = self.layers
        if len(arch) > len(layers):
            layers.extend([None] * (len(arch) - len(layers)))
            for i in range(len(arch)-1, len(layers)-1, -1):
                layer = CIBPlayer(self, i, arch[i])
                layers[i] = layer
        else:
            self.layers = layers[:len(arch)]
            for i in range(len(arch)-1, 0, -1):
                self.layers[i].resize(arch[i])

    def activate(self, layer, next_u=[], baseline=None, delta=lambda WZ: 0):
        res = self.net.biases[layer]
        if layer < len(self.layers):
            if len(next_u) == 0:
                next_u = self.layers[layer+1].u
            WZ = self.net.W[layer+1] * self.net.Z[layer+1]
            res += np.dot(WZ, next_u)
        return res

    def sample(self):
        for (li, layer) in enumerate(self.layers):
            if li == 0:
                continue
            layer.sample()


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
        self.gammas[layer].resize(arch[layer], refcheck=False)
        self.vs[layer].resize(arch[layer], refcheck=False)
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

    def sample_prior_gamma(self, K, layer):
        return npr.randn(K) * np.sqrt(1.0 / self.rho_gamma(layer)) + self.mu_gamma(layer)

    def sample_prior_W(self, K, Kp, layer):
        return npr.randn(K, Kp) * np.sqrt(1.0 / self.rho_w(layer)) + self.mu_w(layer)

    def sample_prior_v(self, K, layer):
        return npr.gamma(self.a(layer), self.b(layer), K)

    def sample_hiddens(sef, hiddens):
        for hidden in hiddens:
            hidden.sample()

    def sample_weights(self, hiddens):
        for l in range(1, self.num_layer):
            W = self.Ws[l]
            WZ = W * self.Zs[l]
            mu_w = self.mu_w(l)
            rho_w = self.rho_w(l)

            for kp in range(arch[l]):
                A = 0
                B = 0
                for hidden in hiddens:
                    prev_u = hidden.layers[l-1].u
                    prev_y = hidden.layers[l-1].y
                    u = hidden.layers[l].u
                    A += u[kp] * (inv_sigmoid(prev_u) - prev_y + WZ[:, kp])
                    B += u[kp]**2
                A *= self.vs[l-1]
                B *= self.vs[l-1]
                rho_w_post = rho_w + B
                std_w_post = np.sqrt(1.0/rho_w_post)
                mu_w_post = (rho_w * mu_w + A) / rho_w_post
                W[:, kp] = npr.randn(arch[l]) * std_w_post + mu_w_post

    def smale_biases(self, hiddens):
        N = len(hiddens)
        arch = self.arch
        for l in range(self.num_layer):
            W = self.Ws[l]
            gamma = self.gammas[l]
            v = self.vs[l]
            mu_gamma = self.mu_gamma(l)
            rho_gamma = self.rho_gamma(l)
            A = 0
            for hidden in hiddens:
                u = hidden.layers[l].u
                y = hidden.layers[l].y
                A += inv_sigmoid(u) - y + gamma
            A *= v
            rho_gamma_post = rho_gamma + N * v
            std_b_post = np.sqrt(1.0/rho_gamma_post)
            mu_gamma_post = (rho_gamma * mu_gamma + A) / rho_gamma_post
            gamma[:] = npr.randn(arch[l]) * std_b_post + mu_gamma_post

    def sample_activation_variance(self, hiddens):
        N = len(hiddens)
        arch = self.arch
        for l in range(self.num_layer):
            v = self.vs[l]
            A = np.zeros(arch[l])
            for hidden in hiddens:
                u = hidden.layers[l].u
                y = hidden.layers[l].y
                A += (inv_sigmoid(u) - y)**2
            a_post = np.ones(arch[l]) + self.a(l) + N / 2.0
            b_post = self.b(l) + A * .5
            self.vs[l] = map(npr.gamma, a_post, b_post)

    def sample_structure(self, hiddens):
        arch = self.arch
        for l in range(self.num_layer-1):
            lp = l+1
            Z = self.Zs[lp]
            W = self.Ws[lp]
            WZ = W * Z
            for k in range(arch[l]):
                singleton = []
                for kp in range(arch[lp]):
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
                            next_u = hidden.layers[lp].u
                            y = hidden.layers[l].y
                            baseline0 = y[k]
                            baseline0 -= WZ[k, kp] * next_u[kp]     # Z = 0.
                            baseline1 = baseline + W[k, kp] * next_u[kp]    # Z = 1.
                            lhood0 += hidden.layers[l].proposal_lhood_y(baseline0, k)
                            lhood1 += hidden.layers[l].proposal_lhood_y(baseline1, k)
                        lhood0 = lhood0 - max(lhood0, lhood1)
                        lhood1 = lhood1 - max(lhood0, lhood1)
                        prob = np.exp(lhood1) / (np.exp(lhood0) + np.exp(lhood1))
                        if npr.rand() < prob:
                            Z[k, kp] = 1
                        else:
                            Z[k, kp] = 0

                # Phase II. remove or add singleton parents.
                Ko = len(singleton)
                if npr.rand() < .5:     # birth.
                    old_arch = list(self.arch)
                    j = old_arch[lp]
                    self.sample_prior_cibp(lp)
                    lhood = np.log(self.alpha(l) * self.beta(l)) - \
                        2 * np.log(Ko+1) - np.log(self.beta(l) + self.arch[l] - 1)
                    for hidden in hiddens:
                        hidden.resize(arch)
                        next_u = hidden.layers[lp].u
                        y = hidden.layers[l].y
                        baseline0 = y[k]
                        bsaeline1 = y[k] + W[k, j] * next_u[j]
                        lhood -= hidden.layers[l].proposal_lhood_y(baseline0, k)
                        lhood += hidden.layers[l].proposal_lhood_y(baseline1, k)
                    if npr.rand() >= np.exp(lhood):  # reject, roll back changes.
                        self.arch = old_arch
                        for hidden in hiddens:
                            hidden.resize(arch)
                else:   # death.
                    ind = npr.choice(Ko)
                    lhood = 2 * np.log(Ko) + np.log(self.beta(l) + self.arch[l] - 1) \
                        - np.log(self.alpha(l)) - np.log(self.beta(l))
                    for hidden in hiddens:
                        next_u = hidden.layers[lp].u
                        y = hidden.layers[l].y
                        baseline0 = y[k] - W[k, ind] * Z[k, ind] * next_u[ind]
                        baseline1 = y[k]
                        lhood += hidden.layers[l].proposal_lhood_y(baseline0, k)
                        lhood -= hidden.layers[l].proposal_lhood_y(baseline1, k)
                    if npr.rand() < np.exp(lhood):  # accept, remove edge.
                        Z[k, ind] = 0

    def train(self, examples, num_iter):
        hiddens = []
        for ex in examples:
            hidden = Hidden(self, ex)
            hiddens.append(hidden)
        for it in range(num_iter):
            self.sample_hiddens(hiddens)
            self.sample_weights(hiddens)
            self.sample_biases(hiddens)
            self.sample_activation_variance(hiddens)
            self.sample_structure(hiddens)