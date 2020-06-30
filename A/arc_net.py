import torch.nn as nn
import torch
import torch.nn.functional as F


class Arc_Net(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim), requires_grad=True)

    def forward(self, feature, m=1):
        x = F.normalize(feature, dim=1)
        w = F.normalize(self.W, dim=0)

        cosa = torch.matmul(x, w) / (torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2))))

        s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax


if __name__ == '__main__':
    arc = Arc_Net(2, 10)
    feature = torch.randn(100, 2)
    out = arc(feature)
    print(out.shape)
