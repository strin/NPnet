import unittest
import numpy as np
import numpy.random as npr
from cibp import *


class TestCIBP(unittest.TestCase):
    def setUp(self):
        pass

    def test_cibp_prior(self):
        npr.seed(0)
        while True:
            net = CIBPnet(alpha=lambda l: 10, beta=lambda l: 1, nv=100)
            if len(net.arch) > 1:
                break
        arch = net.arch[1]
        net.sample_prior_cibp(1)
        new_arch = net.arch[1]
        self.assertEqual(arch + 1, new_arch)
        self.assertEqual(arch + 1, net.Ws[1].shape[1])
        


        



if __name__ == "__main__":
    unittest.main()
