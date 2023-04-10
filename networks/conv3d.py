'''
Implementation of 3D conv layer for abstract grids
'''
import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F
from torch.autograd import Function
from backend import _backend, _backend_context
from gridencoder.grid import grid_encode
import time
from tqdm import tqdm
import numpy as np

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

abstractConv3DFunction = _abstract_conv3d.apply

class HashRouterLayer(nn.Module):
    # combines the hash table routing with a mlp for querying per-pixel values
    def __init__(self, resolutions, offsets, num_levels=16, log_hashmap_size=19, embed_channels=2, mlp_channels=[32, 32], out_channels=1, activ=nn.LeakyReLU()):
        super().__init__()
        self.resolutions = resolutions
        self.offsets = offsets
        self.num_levels = num_levels
        self.log_hashmap_size = log_hashmap_size
        self.embed_channels = embed_channels
        # get mlp
        mlp = []
        embed_channels = embed_channels * num_levels
        for c in mlp_channels:
            mlp.append(nn.Linear(embed_channels, c))
            embed_channels = c
            mlp.append(activ)
        mlp.append(nn.Linear(embed_channels, out_channels))
        self.mlp = nn.Sequential(*mlp)
        self.base_resolution = base_resolution = resolutions[0].item()
        self.desired_resolution = desired_resolution = resolutions[-1].item()
        self.per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        # Done
        pass
    
    def forward(self, inputcoords, embeddings, bound=1):
        # input_coords: [*, batch_size, 3]
        # embeddings: [num_embeddings, batch_size, embed_channels]
        # outputs: [*, batch_size, output_channels]
        batch_size = inputcoords.shape[-2]
        prefix_shape = list(inputcoords.shape[:-2])
        # if batch size is 1, just squeeze the batch size dimension and use that
        if batch_size == 1:
            input = (inputcoords + bound)/(2*bound)
            input = input.view(-1, 3)
            outputs = grid_encode(input, embeddings[:, 0], self.offsets, self.per_level_scale, self.base_resolution, inputcoords.requires_grad, 1, True, 0)
            outputs = outputs.view(prefix_shape + [1, self.num_levels * self.embed_channels])
            alloutputs = self.mlp(outputs)
        else:
            alloutputs = []
            for b in range(batch_size):
                input = (inputcoords[..., b, :] + bound)/(2*bound)
                input = input.view(-1, 3)
                # last three parameters are: gridtype, align_corners, mode
                outputs = grid_encode(input, embeddings[:, b].contiguous(), self.offsets, self.per_level_scale, self.base_resolution, inputcoords.requires_grad, 1, True, 0)
                outputs = outputs.view(prefix_shape + [1, self.num_levels * self.embed_channels])
                alloutputs.append(outputs)
            alloutputs = torch.cat(alloutputs, dim=-2)
            alloutputs = self.mlp(alloutputs)
        return alloutputs

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
        self.register_parameter('weight', nn.Parameter(0.01*torch.randn(num_levels, *kernel_size, channels_in, channels_out)))   # keep channels_in at the end to 
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_levels, channels_out)) if bias else None)

    def forward(self, input):
        ''' forward pass 
        input: (B, N, C_in)
        '''
        return abstractConv3DFunction(input, self.offsets, self.resolutions, self.weight, self.bias, self.num_levels, self.hashmap_size)

## Check this
if __name__ == '__main__':
    import gridencoder as ge
    L = 19
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None].contiguous() * 1e3  # [1, N, 2]
    embed = embed.detach()
    print(embed.min(), embed.max())
    # embed = embed.expand(4, -1, -1).contiguous()
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    print(embed.shape, resolutions.shape, offsets.shape)

    router = HashRouterLayer(resolutions, offsets, 16, L, 2, [32, 32], out_channels=3).cuda()
    inputs = torch.rand(10000, 1, 3).cuda()*2 - 1
    print(router)
    a = time.time()
    y = router(inputs, embed)
    print(time.time() - a)
    print(inputs.shape, embed.shape, y.shape)

    # get a ground truth
    # encoder2 = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L, level_dim=4).cuda()
    # gt = encoder2.embeddings[:, None].contiguous() * 1 + 1 # [1, N, 2]
    # gt = gt.detach()

    ### Deep network
    # seq = [4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 4]
    # f = 2
    # module = []
    # for s in seq:
    #     module.append(AbstractConv3D(f, s, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda())
    #     module.append(nn.LeakyReLU())
    #     f = s
    # module = nn.Sequential(*module[:-1])
    # print(module)
    # optim = torch.optim.Adam(module.parameters(), lr=1e-3)
    # for i in range(2000):
    #     optim.zero_grad()
    #     out = module(embed)
    #     # loss = F.binary_cross_entropy_with_logits(out, gt)
    #     loss = F.mse_loss(out, gt)
    #     loss.backward()
    #     optim.step()
    #     print(loss.item())

    # layer = AbstractConv3D(2, 8, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    # layer2 = AbstractConv3D(8, 8, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()

    # a = time.time()
    # output = abstractContextFunction(gt, offsets, resolutions, 16, 2**L)
    # print(time.time() - a)
    # print("done.")
    # #print(output)
    # # print(output.min(), output.max(), output.shape, embed.min(), embed.max(), embed.shape)
    # #input()

    # # compute time
    # # embed = embed.expand(-1, 32, -1).contiguous()
    # a = time.time()
    # output = layer(embed)
    # print(time.time() - a)
    # print(output.min(), output.max(), output.shape)

    # optim = torch.optim.Adam(list(layer.parameters()) + list(layer2.parameters()), lr=5e-2)
    # pbar = tqdm(range(200))
    # for it in pbar:
    #     optim.zero_grad()
    #     out = layer(embed) 
    #     output = layer2(F.leaky_relu(out))
    #     loss = 0
    #     # loss = F.binary_cross_entropy_with_logits(output, output.detach()*0 + 1)
    #     for i in range(16):
    #         sz = int(min(encoder.max_params, resolutions[i]**3))
    #         # loss += ((output[offsets[i]:offsets[i]+sz] - 1)**2).mean()
    #         ### cross-entropy loss shows very good performance (with Adam, lr = 5e-2)
    #         loss += F.binary_cross_entropy_with_logits(output[offsets[i]:offsets[i]+sz], 0*output[offsets[i]:offsets[i]+sz].detach()+1)
    #     loss = loss / 16.0
    #     (loss).backward()
    #     # loss = ((output - 1)**2).mean() 
    #     # loss.backward()
    #     optim.step()
    #     pbar.set_description("iter: %d, loss: %.6f" % (it, loss.item()))
    # print(layer2.weight.abs().mean(), layer2.bias)
    # print(layer.weight.abs().mean(), layer.bias)
    # print(output)

    # optim = torch.optim.SGD(layer.parameters(), lr=1e-0, momentum=0.9)
    # optim = torch.optim.Adam(list(layer.parameters()), lr=4e-2)
    # pbar = tqdm(range(300))
    # for it in pbar:
    #     optim.zero_grad()
    #     output = layer(embed)
    #     loss = 0
    #     for i in range(16):
    #         sz = int(min(encoder.max_params, resolutions[i]**3))
    #         loss += ((output[offsets[i]:offsets[i]+sz] - 1)**2).mean()
    #     loss.backward()
    #     optim.step()
    #     pbar.set_description("iter: %d, loss: %.4f" % (it, loss.item()))
    # print(layer.weight.abs().mean(), layer.bias)
    # print(output)

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
