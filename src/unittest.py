import torch
from torch import nn
import time

layer = nn.Conv3d(2, 4, 3, bias=True, padding=1).cuda()
layer2 = nn.Conv3d(4, 8, 3, bias=True, padding=1).cuda()
inp = torch.rand(1, 2, 256, 256, 256).cuda()

a = time.time()
out = layer(inp)
print(time.time() - a)
print(out.shape)

a = time.time()
out = layer(inp)
print(time.time() - a)
print(out.shape)

a = time.time()
out = layer2(out)
print(time.time() - a)
print(out.shape)