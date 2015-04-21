from cibp import *

if __name__ == "__main__":
    net = CIBPnet(alpha=lambda l: 10, beta=lambda l: 1, nv=100)
    print net.arch
