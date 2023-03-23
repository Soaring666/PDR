import torch

a = torch.randn(3, 4, 5)
c = a.shape
d = torch.randn(c)
print(a.shape)
print(d.shape)