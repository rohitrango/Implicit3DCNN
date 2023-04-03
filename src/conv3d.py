'''
Implementation of 3D conv layer for abstract grids
'''
import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F
from torch.autograd import Function
from backend import _backend
import time
from tqdm import tqdm

class _abstract_conv3d(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, offsets, resolutions, weight, bias, num_levels, hashmap_size):
        ''' Forward pass

        :input: [batch_size, num_embeddings, channels_in]
        :weight: [num_levels, kernel_size, kernel_size, kernel_size, channels_in, channels_out]
        :bias: [num_levels, channels_out]
        '''
        ctx.save_for_backward(input, offsets, resolutions, weight, bias, torch.tensor(num_levels), torch.tensor(hashmap_size), torch.tensor(input.requires_grad))
        batch_size, num_embedding = input.shape[:2]
        channels_out = weight.shape[-1]
        output = torch.zeros((batch_size, num_embedding, channels_out), device=input.device, dtype=input.dtype)
        # Save backward tensors and return output
        output = _backend.abstract_conv3d_forward(input, output, offsets, resolutions, weight, bias, num_levels, hashmap_size)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        ''' Backward pass

        :grad_outputs: [batch_size, num_embeddings, channels_out]
        '''
        input, offsets, resolutions, weight, bias, num_levels, hashmap_size, inp_requires_grad = ctx.saved_tensors
        num_levels = int(num_levels.item())
        hashmap_size = int(hashmap_size.item())
        inp_requires_grad = bool(inp_requires_grad.item())
        # get outputs
        input_grad = torch.zeros_like(input, dtype=input.dtype, device=input.device)
        weight_grad = torch.zeros_like(weight, dtype=weight.dtype, device=weight.device)
        bias_grad   = torch.zeros_like(bias, dtype=bias.dtype, device=bias.device) if bias is not None else None
        # a = time.time()
        input_grad, weight_grad, bias_grad = _backend.abstract_conv3d_backward(grad_outputs, input_grad, weight_grad, bias_grad, \
                                                                               inp_requires_grad, input, offsets, resolutions, weight, bias, num_levels, hashmap_size)
        # print("backward time: ", time.time() - a)
        # backward pass
        return input_grad, None, None, weight_grad, bias_grad, None, None


class AbstractConv3D(nn.Module):
    ''' Actual nn Module that implements the 3D convolution layer'''
    def __init__(self, channels_in, channels_out, resolutions, offsets, kernel_size, bias=True, num_levels=16,
                 log_hashmap_size=19):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        # hash encoding params
        self.num_levels = num_levels
        self.hashmap_size = int(2**log_hashmap_size)
        # load weights
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 3, "kernel_size should be int or tuple/list of length 3"
        ## list of grid sizes/resolutions, and offsets, should be int32
        self.register_buffer('resolutions', resolutions.clone().detach().int())
        self.register_buffer('offsets', offsets.clone().detach().int())
        ### load weights now
        self.register_parameter('weight', nn.Parameter(0.1*torch.randn(num_levels, *kernel_size, channels_in, channels_out)))   # keep channels_in at the end to 
        self.register_parameter('bias', nn.Parameter(0.01*torch.randn(num_levels, channels_out)) if bias else None)

    def forward(self, input):
        ''' forward pass 
        input: (B, N, C_in)
        '''
        return _abstract_conv3d.apply(input, self.offsets, self.resolutions, self.weight, self.bias, self.num_levels, self.hashmap_size)

## Check this
if __name__ == '__main__':
    import gridencoder as ge
    L = 19
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None].contiguous() * 1e4  # [1, N, 2]
    embed = embed.detach()
    print(embed.min(), embed.max())
    # embed = embed.expand(4, -1, -1).contiguous()
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    print(embed.shape, resolutions.shape, offsets.shape)

    # get a ground truth
    encoder2 = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L, level_dim=4).cuda()
    gt = encoder2.embeddings[:, None].contiguous() * 1e3 + 1  # [1, N, 2]
    gt = gt.detach()

    layer = AbstractConv3D(2, 8, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    layer2 = AbstractConv3D(8, 4, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()

    # compute time
    # embed = embed.expand(-1, 32, -1).contiguous()
    a = time.time()
    output = layer(embed)
    print(time.time() - a)
    print(output.min(), output.max(), output.shape)

    # optim = torch.optim.AdamW(list(layer.parameters()) + list(layer2.parameters()), lr=1e-2)
    # pbar = tqdm(range(300))
    # for it in pbar:
    #     optim.zero_grad()
    #     out = layer(embed) 
    #     output = layer2(F.leaky_relu(out))
    #     # output = layer(embed)
    #     loss = ((output - 1)**2).mean() 
    #     loss.backward()
    #     optim.step()
    #     pbar.set_description("iter: %d, loss: %.4f" % (it, loss.item()))
    # print(layer2.weight.abs().mean(), layer2.bias)
    # print(output)

    # optim = torch.optim.AdamW(list(layer.parameters()), lr=4e-3)
    optim = torch.optim.SGD(layer.parameters(), lr=1e-0, momentum=0.9)
    pbar = tqdm(range(300))
    for it in pbar:
        optim.zero_grad()
        output = layer(embed)
        loss = ((output - 1)**2).mean() 
        loss.backward()
        optim.step()
        pbar.set_description("iter: %d, loss: %.4f" % (it, loss.item()))
    print(layer.weight.abs().mean(), layer.bias)
    print(output)

    # inp = torch.randn(1, 32, 128, 128, 128).cuda()
    # conv = nn.Conv3d(32, 32, 3, padding=1).cuda()
    # optim = torch.optim.SGD(conv.parameters(), lr=5e-1)
    # pbar = tqdm(range(1000))
    # for i in pbar:
    #     optim.zero_grad()
    #     output = conv(inp)
    #     loss = ((output - 1)**2).mean()
    #     loss.backward()
    #     optim.step()
    #     pbar.set_description("iter: %d, loss: %.4f" % (i, loss.item()))

    # layer2 = AbstractConv3D(4, 32, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    # a = time.time()
    # output = layer2(output)
    # print(time.time() - a)
    # print(output.min(), output.max(), output.shape)

    # layer = AbstractConv3D(32, 2, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    # a = time.time()
    # output = layer(output)
    # print(time.time() - a)
    # print(output.min(), output.max(), output.shape)
