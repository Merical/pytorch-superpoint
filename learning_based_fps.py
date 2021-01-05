from models.SuperPointNetOpen import *
from models.ShufflePointNet import *
from models.GCNv2 import *
from models.D2Net import *

import time

size = [960, 1280]

with torch.no_grad():
    input = torch.rand([1, 1, size[0], size[1]]).cuda()

    model = SuperPointNetOpen().cuda()
    model.eval()
    output = model.forward(input)

    tic = time.time()
    for _ in range(100):
        output = model.forward(input)
    toc = time.time()
    print("Forward Flow Done, Superpoint cost {} ms".format((toc - tic) * 1000 / 100))
    del model

    model = ShufflePointNet().cuda()
    model.eval()
    output = model.forward(input)

    tic = time.time()
    for _ in range(100):
        output = model.forward(input)
    toc = time.time()
    print("Forward Flow Done, ShufflePointNet cost {} ms".format((toc - tic) * 1000 / 100))
    del model

    model = GCNv2().cuda()
    model.eval()
    output = model.forward(input)

    tic = time.time()
    for _ in range(100):
        output = model.forward(input)
    toc = time.time()
    print("Forward Flow Done, GCNv2 cost {} ms".format((toc - tic) * 1000 / 100))
    del model

    del input
    input = torch.rand([1, 3, size[0], size[1]]).cuda()

    model = D2Net().cuda()
    model.eval()
    output = model.forward(input)

    tic = time.time()
    for _ in range(100):
        output = model.forward(input)
    toc = time.time()
    print("Forward Flow Done, D2Net cost {} ms".format((toc - tic) * 1000 / 100))
    del model