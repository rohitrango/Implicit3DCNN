''' Implementation of the tensoRF network '''
import torch
import torch.nn as nn
from torch.nn import functional as F

class TensoRFEncoder(nn.Module):
    ''' class that defines tensoRF style features to encode matrices into features '''
    def __init__(self, image_size, num_features, agg_method='sum'):
        super().__init__()
        self.image_size = image_size
        x, y, z = image_size
        # define line and matrix features (the sizes should be indexed [Y, X] because coordinates will be of the form [x, y])
        self.mat_features = nn.ParameterList([nn.Parameter(torch.randn(1, num_features, dim2, dim1)) for dim1, dim2 in [(y, z), (z, x), (x, y)]])
        self.line_features = nn.ParameterList([nn.Parameter(torch.randn(1, num_features, dim, 1)) for dim in image_size])
        self.agg_method = agg_method
        assert agg_method in ['sum', 'cat']
        
    def forward(self, coords):
        ''' assume coords to be of size [B, N, 3] normalized from [-1, 1] '''
        xy, yz, zx = coords[..., None, [0, 1]], coords[..., None, [1, 2]], coords[..., None, [2, 0]]  # [B, N, 1, 2]
        x, y, z = [coords[..., None, i:i+1] for i in range(3)]  # [B, N, 1, 1]
        x, y, z = [torch.cat([torch.zeros_like(t), t], dim=-1) for t in [x, y, z]] # [B, N, 1, 2]
        # TODO: Weird bug where grid_sample takes much longer with align_corners=True
        align_corners = False
        mat_features = [F.grid_sample(matfeat, t, align_corners=align_corners) for matfeat, t in zip(self.mat_features, [yz, zx, xy])]
        line_features = [F.grid_sample(linefeat, t, align_corners=align_corners) for linefeat, t in zip(self.line_features, [x, y, z])]
        feats = [f1 * f2 for f1, f2 in zip(mat_features, line_features)]
        if self.agg_method == 'sum':
            return torch.stack(feats, dim=0).sum(0).squeeze(-1).permute(0, 2, 1)  # [B, N, C]
        else:
            return torch.cat(feats, dim=1).squeeze(-1).permute(0, 2, 1)  # [B, N, 3C]


class TensoRFConv(nn.Module):
    ''' Wrapper around convolution block that applies convolution on tensoRF features directly '''
    def __init__(self, input_features, output_features, kernel_size=3,):
        super().__init__()
        padding_size = kernel_size // 2
        self.line_convs = nn.ModuleList([nn.Conv2d(input_features, output_features, kernel_size=(kernel_size, 1), padding=(padding_size, 0)) for _ in range(3)])
        self.mat_convs = nn.ModuleList([nn.Conv2d(input_features, output_features, kernel_size=(kernel_size, kernel_size), padding=padding_size) for _ in range(3)])

    def forward(self, line_features, mat_features):
        ''' line_features: [B, C, D, 1] and mat_features = [B, C, D, D] '''
        line_outputs = [conv(line_features[i]) for i, conv in enumerate(self.line_convs)]
        mat_outputs  = [conv(mat_features[i]) for i, conv in enumerate(self.mat_convs)]
        return line_outputs, mat_outputs



if __name__ == '__main__':
    net = TensoRFEncoder((240, 155, 155), 16).cuda()
    print(net)
    coords = torch.rand(1, 1000, 3).cuda() * 2 - 1
    from time import time
    a = time()
    out = net(coords)
    print(time() - a)
    convnet = TensoRFConv(16, 64).cuda()
    out = convnet(net.line_features, net.mat_features)
    for o in out:
        for k in o:
            print(k.shape)
        print()