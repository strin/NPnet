# implement basic and multi-try MH samplers.
import numpy as np
import numpy.random as npr


class BasicMH(object):
    def __init__(self, propose, lhood, proposal_lhood=lambda x, old_x: 0):
        self.propose = propose
        self.lhood = lhood
        self.proposal_lhood = proposal_lhood
        self.num_sample = 0
        self.num_acc = 0

    def sample(self, x):
        old_x = x
        x = self.propose(old_x)
        score = min(0, self.lhood(x) - self.lhood(old_x)
                    + self.proposal_lhood(old_x, x) - self.proposal_lhood(x, old_x))
        coin = npr.rand()
        self.num_sample += 1
        if np.log(coin) < score:
            self.num_acc += 1
            return x
        else:
            return old_x

    def acc_rate(self):
        return float(self.num_acc) / self.num_sample
