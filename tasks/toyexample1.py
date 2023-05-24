'''
author: rohitrango

Script to generate toy examples where the volumes contain a ball at some locations.

- The segmentation task is to segment all the balls that have another ball at diagonally opposite location. 
- The classification task is to classify the volume from 1 to 8 numbers
'''
import numpy as np
from scipy.ndimage import gaussian_filter
from os import path as osp
from tqdm import tqdm
from glob import glob
import gridencoder as ge
import torch
from torch import nn
from torch.nn import functional as F
from networks.conv3d import AbstractConv3D, HashRouterLayer
from networks.contextlayer import AbstractContextLayer

low_thr = 0.6
hi_thr = 0.75

class Resblock(nn.Module):
    def __init__(self, input_channels, output_channels, resolutions, offsets, log_hashmap_size, num_levels):
        super().__init__()
        self.conv1 = AbstractConv3D(input_channels, output_channels, resolutions, offsets, 3, True, num_levels=num_levels, log_hashmap_size=log_hashmap_size)
        self.conv2 = AbstractConv3D(output_channels, output_channels, resolutions, offsets, 3, True, num_levels=num_levels, log_hashmap_size=log_hashmap_size)
        # self.context = AbstractContextLayer(input_channels, output_channels, resolutions, offsets, affine=True, num_levels=num_levels, log_hashmap_size=log_hashmap_size)
        self.context = nn.Linear(input_channels, output_channels)

    def forward(self, inp):
        x = F.leaky_relu(self.conv1(inp))
        x = F.leaky_relu(self.conv2(x))
        x = x + self.context(inp)
        return x

save_path = "/data/rohitrango/Implicit3DCNNTasks/toyexample1"

def generate_data(num_images=300):
    ''' Function to generate data '''
    volume_size = 256
    sigma = 30
    # get labels file
    labels_file = open(osp.join(save_path, "labels.txt"), "w")
    for i in tqdm(range(num_images)):
        volume = np.zeros((volume_size, volume_size, volume_size))       
        loc = np.random.randint(4)  # we will multiply with 2 to get 8 different choices
        # find radius
        coord = np.random.randint(10, size=(3,))-5 + np.array([volume_size//4, volume_size//4, volume_size//4])
        if loc > 0:
            coord[loc-1] += volume_size//2
        # flip with half probability
        if np.random.rand() > 0.5:
            coord = volume_size - coord
        # put the ball there
        x, y, z = coord
        volume[x, y, z] = (np.sqrt(2*np.pi)*sigma)**3
        mirror = np.random.randint(2)
        if mirror:
            coordnew = volume_size - coord
            x, y, z = coordnew
            volume[x, y, z] = (np.sqrt(2*np.pi)*sigma)**3
        # smear the ball
        volume = gaussian_filter(volume, sigma=sigma).astype(np.float32)
        volume[volume < low_thr] = 0
        # volume = volume / volume.max()
        segm = (volume >= hi_thr).astype(np.uint8) * mirror
        # classification
        label = loc + 4*mirror
        # save them
        np.save(osp.join(save_path, f"img_{i}.npy"), volume)
        np.save(osp.join(save_path, f"seg_{i}.npy"), segm)
        labels_file.write("{}\n".format(label))
    labels_file.close()

def learn_features(num_images=300):
    ''' Use this block of code to learn the features (and save them in same directory) '''
    train_images = int(num_images * 0.75)
    val_images = num_images - train_images
    print(f"We got {train_images} train and {val_images} val images")
    # select first n images
    images = sorted(glob(osp.join(save_path, "img_*.npy")), key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))[:train_images]
    encoders = [ge.GridEncoder(desired_resolution=128, gridtype='tiled', align_corners=True, log2_hashmap_size=19).cuda() for _ in images ]
    encoder_optims = [torch.optim.Adam(encoder.parameters(), lr=1e-3) for encoder in encoders]
    # decoder
    decoder = nn.Sequential(nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 1)).cuda()
    decoder_optims = torch.optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=1e-5)
    # run optimization
    num_iters_per_img = 120
    pbar = tqdm(range(num_iters_per_img*train_images))
    lossitem = 0
    for it in pbar:
        imgid = it%train_images
        if imgid == 0:
            lossitem = 0
        encoder_optims[imgid].zero_grad()
        decoder_optims.zero_grad()
        # load image
        img = np.load(images[imgid])
        xyz = np.random.randint(0, 256, size=(20000, 3))
        imgpoints = img[xyz[:, 0], xyz[:, 1], xyz[:, 2]]  # [B, ]
        imgpoints = torch.from_numpy(imgpoints).cuda().float()[:, None]   # [B, 1]
        xyz = torch.from_numpy(xyz).cuda().float()/255.0*2 - 1            # [B, 3] valued from [-1, 1] inclusive
        # run prediction
        pred_img = decoder(encoders[imgid](xyz))
        loss = F.mse_loss(pred_img, imgpoints)
        loss.backward()
        # append it to lossitem
        lossitem = (lossitem*imgid + loss.item())/(imgid+1)
        # optim gradients
        encoder_optims[imgid].step()
        decoder_optims.step()
        pbar.set_description(f"Loss: {loss.item():.7f}, lossitem: {lossitem:.7f}")
    ## save all encoded features
    for i, encoder in enumerate(encoders):
        torch.save(encoder.state_dict(), osp.join(save_path, f"encoded{i}.pth"))
    ## save decoder
    torch.save(decoder.state_dict(), osp.join(save_path, f"decoder.pth"))

@torch.no_grad()
def check_training_images(num_images=300, train=True):
    ''' checking quality of learnt training images '''
    train_images = int(num_images * 0.75)
    val_images = num_images - train_images
    print(f"We got {train_images} train and {val_images} val images")
    # select first n images
    images = sorted(glob(osp.join(save_path, "img_*.npy")), key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
    if train:
        print("Selecting train images for checking psnr.")
        images = images[:train_images]
    else:
        print("Selecting val images for checking psnr.")
        images = images[train_images:]

    encoder = ge.GridEncoder(desired_resolution=128, gridtype='tiled', align_corners=True, log2_hashmap_size=19).cuda()
    # decoder
    decoder = nn.Sequential(nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 1)).cuda()
    decoder.load_state_dict(torch.load(osp.join(save_path, f"decoder.pth")))
    # for each image, compute psnr 
    psnrs = []
    pbar = tqdm(range(train_images)) if train else tqdm(range(val_images))
    start = 0 if train else train_images
    for i in pbar:
        encoder.load_state_dict(torch.load(osp.join(save_path, f"encoded{i+start}.pth")))
        # compute xyz
        x, y, z = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij')  
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        xyz = torch.from_numpy(xyz).cuda().float()/255.0*2 - 1            # [B, 3] valued from [-1, 1] inclusive 
        pred = decoder(encoder(xyz)).cpu().numpy().reshape(256, 256, 256)
        mse = np.mean((pred - np.load(images[i]))**2)
        psnr = 10*np.log10(1/mse)
        psnrs.append(psnr)
        pbar.set_description(f"PSNR: {psnr:.4f}")
    print("Mean PSNR: ", np.mean(psnrs))
    print("Std PSNR: ", np.std(psnrs))
    print("Min PSNR: {:4f}, Max PSNR: {:4f}".format(np.min(psnrs), np.max(psnrs)))

def learn_validation_features(num_images=300):
    ''' once training features are learnt, learn the encoders for validation images '''
    train_images = int(num_images * 0.75)
    val_images = num_images - train_images
    print(f"We got {train_images} train and {val_images} val images")
    # select first n images
    valimages = sorted(glob(osp.join(save_path, "img_*.npy")), key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))[train_images:]
    encoders = [ge.GridEncoder(desired_resolution=128, gridtype='tiled', align_corners=True, log2_hashmap_size=19).cuda() for _ in valimages]
    encoder_optims = [torch.optim.Adam(encoder.parameters(), lr=1e-3) for encoder in encoders]
    # decoder
    decoder = nn.Sequential(nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 1)).cuda()
    decoder.load_state_dict(torch.load(osp.join(save_path, f"decoder.pth")))
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    # run optimization
    num_iters_per_img = 120
    pbar = tqdm(range(num_iters_per_img*val_images))
    lossitem = 0
    for it in pbar:
        imgid = it%val_images
        if imgid == 0:
            lossitem = 0
        encoder_optims[imgid].zero_grad()
        # load image
        img = np.load(valimages[imgid])
        xyz = np.random.randint(0, 256, size=(20000, 3))
        imgpoints = img[xyz[:, 0], xyz[:, 1], xyz[:, 2]]  # [B, ]
        imgpoints = torch.from_numpy(imgpoints).cuda().float()[:, None]   # [B, 1]
        xyz = torch.from_numpy(xyz).cuda().float()/255.0*2 - 1            # [B, 3] valued from [-1, 1] inclusive
        # run prediction
        pred_img = decoder(encoders[imgid](xyz))
        loss = F.mse_loss(pred_img, imgpoints)
        loss.backward()
        # append it to lossitem
        lossitem = (lossitem*imgid + loss.item())/(imgid+1)
        # optim gradients
        encoder_optims[imgid].step()
        pbar.set_description(f"Loss: {loss.item():.7f}, lossitem: {lossitem:.7f}")
    ## save all encoded features
    for i, encoder in enumerate(encoders):
        torch.save(encoder.state_dict(), osp.join(save_path, f"encoded{i+train_images}.pth"))
    ## save decoder
    torch.save(decoder.state_dict(), osp.join(save_path, f"decoder.pth"))

def train_classifier(num_images=300, EPOCHS=100):
    # load all the encoded features
    encoder = ge.GridEncoder(desired_resolution=128, gridtype='tiled', align_corners=True, log2_hashmap_size=19).cuda()
    train_images = int(num_images * 0.75)
    val_images = num_images - train_images
    encoders = sorted(glob(osp.join(save_path, "encoded*.pth")), key=lambda x: int(x.split('/')[-1].split('.')[0].split('d')[-1]))
    trainencoders = encoders[:train_images]
    valencoders = encoders[train_images:]
    # load labels
    with open(osp.join(save_path, "labels.txt"), 'r') as f:
        labels = f.read().split('\n')[:-1]
        labels = [int(label) for label in labels]
        print(len(labels))
    trainlabels = labels[:train_images]
    vallabels = labels[train_images:]

    # init network
    resolutions, offsets = encoder.resolutions, encoder.offsets
    net = []
    net.append(Resblock(2, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    net.append(Resblock(8, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    net.append(Resblock(8, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    # net.append(Resblock(8, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    net = nn.Sequential(*net).cuda()
    print(net)
    classifier = nn.Linear(8*16, 8).cuda()
    def forward_pass(net, classifier, x):
        # x : [N, B, 2]
        x = net(x)
        l = []
        for i in range(encoder.num_levels):
            sz = min(encoder.resolutions[i]**3, 2**encoder.log2_hashmap_size)
            l.append(x[offsets[i]:offsets[i]+sz].mean(0))
        x = torch.cat(l, dim=1) # [B, C*16]
        x = classifier(x)
        return x

    optim = torch.optim.Adam(list(net.parameters()) + list(classifier.parameters()), lr=5e-4, weight_decay=1e-6)
    # load all the encoded features
    for epoch in range(EPOCHS):
        pbar = tqdm(range(train_images))
        losses = []
        accuracy = []
        # optim.zero_grad()
        for i in pbar:
            # load ith
            inp = torch.load(trainencoders[i], map_location='cpu')['embeddings'].cuda().detach()[:, None].contiguous()
            # standard deviation greatly improves performance
            # inp = inp / inp.std()
            inp = inp / 0.05
            out = forward_pass(net, classifier, inp)
            label = torch.tensor([trainlabels[i]]).cuda()
            loss = F.cross_entropy(out, label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.7f}")
            losses.append(loss.item())
            accuracy.append((out.argmax(1) == label).float().mean().item())
        
        ## val images
        pbar = tqdm(range(val_images))
        val_accuracy = []
        for i in pbar:
            inp = torch.load(valencoders[i], map_location='cpu')['embeddings'].cuda().detach()[:, None].contiguous()
            # standard deviation greatly improves performance
            # inp = inp / inp.std()
            inp = inp / 0.05
            out = forward_pass(net, classifier, inp)
            label = torch.tensor([vallabels[i]]).cuda()
            val_accuracy.append((out.argmax(1) == label).float().mean().item())
        print("Epoch: {}, Average loss: {}, train accuracy: {:04f}, val accuracy: {:04f}".format(epoch, np.mean(losses), 100*np.mean(accuracy), 100*np.mean(val_accuracy)))
    
def dice_loss(p, q, val=False):
    # no batch, single image
    return F.binary_cross_entropy_with_logits(p, q)
    #p = torch.sigmoid(p) if not val else (p > 0).float()
    #num = (2*(p*q).mean()) + 1e-4
    #den = p.mean() + q.mean() + 1e-4

    #if val:
        #return num/den
    #else:
        #return 1 - num/den

def train_segmentation(num_images=300, EPOCHS=100):
    encoder = ge.GridEncoder(desired_resolution=128, gridtype='tiled', align_corners=True, log2_hashmap_size=19).cuda()
    train_images = int(num_images * 0.75)
    val_images = num_images - train_images
    encoders = sorted(glob(osp.join(save_path, "encoded*.pth")), key=lambda x: int(x.split('/')[-1].split('.')[0].split('d')[-1]))
    segmentations = sorted(glob(osp.join(save_path, "seg*.npy")), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

    # load labels
    with open(osp.join(save_path, "labels.txt"), 'r') as f:
        labels = f.read().split('\n')[:-1]
        labels = [int(label) for label in labels]
        print(len(labels))
    trainlabels = labels[:train_images]
    vallabels = labels[train_images:]

    # divide into train and val
    trainencoders, trainsegs = encoders[:train_images], segmentations[:train_images]
    valencoders, valsegs = encoders[train_images:], segmentations[train_images:]
    # init network
    resolutions, offsets = encoder.resolutions, encoder.offsets
    net = []
    net.append(Resblock(2, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    net.append(Resblock(8, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    net.append(Resblock(8, 8, resolutions, offsets, encoder.log2_hashmap_size, encoder.num_levels))
    net = nn.Sequential(*net).cuda()
    print(net)
    decoder = HashRouterLayer(resolutions, offsets, encoder.num_levels, encoder.log2_hashmap_size, 8, [], 1).cuda()
    optim = torch.optim.Adam(list(net.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)
    # preload the grid
    x, y, z = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij')  
    xyz = np.stack([x, y, z], axis=-1)                                # [256, 256, 256, 3]     
    N = 2
    xyz = xyz[::N, ::N, ::N]
    print(xyz.shape)
    xyz = torch.from_numpy(xyz).cuda().float()/255.0*2 - 1            # [*, 3] valued from [-1, 1] inclusive 
    xyz = xyz[..., None, :]                                           # [*, 1, 3]
    # load all the encoded features
    for epoch in range(EPOCHS):
        ## val images
        if epoch == 0:
            pbar = tqdm(range(val_images))
            val_loss, vallowloss, valhighloss  = [] , [], []
            with torch.no_grad():
                for i in pbar:
                    inp = torch.load(valencoders[i], map_location='cpu')['embeddings'].cuda().detach()[:, None].contiguous()
                    seg = torch.from_numpy(np.load(valsegs[i])[::N, ::N, ::N]).cuda().float() # [256, 256, 256 ]
                    inp = inp / 0.05
                    embed = net(inp) # [numembeddings, B, C]
                    out = decoder(xyz, embed)[..., 0, 0]  # [256, 256, 256, 1, 1]
                    loss = dice_loss(out, seg, val=True)
                    val_loss.append(loss.item())
                    if vallabels[i] < 4:
                        vallowloss.append(loss.item())
                    else:
                        valhighloss.append(loss.item())
            print("Epoch: {}, val. loss: {:04f}, val. low loss: {:04f}, val. high loss: {:04f}".format(epoch, np.mean(val_loss), np.mean(vallowloss), np.mean(valhighloss)))
            print()

        pbar = tqdm(range(train_images))
        losses = []
        lowlosses, highlosses = [], []
        for i in pbar:
            # load ith
            inp = torch.load(trainencoders[i], map_location='cpu')['embeddings'].cuda().detach()[:, None].contiguous()
            seg = torch.from_numpy(np.load(trainsegs[i])[::N, ::N, ::N]).cuda().float() # [256 **3 ]
            # standard deviation greatly improves performance (inp = inp / inp.std()) or just divide by a reasonable constant
            inp = inp / 0.05
            # forward pass
            embed = net(inp) # [numembeddings, B, C]
            out = decoder(xyz, embed)[..., 0, 0]  # [256, 256, 256, 1, 1]
            loss = F.binary_cross_entropy_with_logits(out, seg) # + 0.1*dice_loss(out, seg)
            # loss = dice_loss(out, seg)
            # out = forward_pass(net, classifier, inp)
            # label = torch.tensor([trainlabels[i]]).cuda()
            # loss = F.cross_entropy(out, label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.7f}, Label: {trainlabels[i]}")
            losses.append(loss.item())
            if trainlabels[i] < 4:
                lowlosses.append(loss.item())
            else:
                highlosses.append(loss.item())
        
        ## val images
        pbar = tqdm(range(val_images))
        val_loss, vallowloss, valhighloss  = [] , [], []
        with torch.no_grad():
            for i in pbar:
                inp = torch.load(valencoders[i], map_location='cpu')['embeddings'].cuda().detach()[:, None].contiguous()
                seg = torch.from_numpy(np.load(valsegs[i])[::N, ::N, ::N]).cuda().float() # [256, 256, 256 ]
                inp = inp / 0.05
                embed = net(inp) # [numembeddings, B, C]
                out = decoder(xyz, embed)[..., 0, 0]  # [256, 256, 256, 1, 1]
                loss = dice_loss(out, seg)
                val_loss.append(loss.item())
                if vallabels[i] < 4:
                    vallowloss.append(loss.item())
                else:
                    valhighloss.append(loss.item())
        print("Epoch: {}, avg. loss: {:04f}, avg. low loss: {:04f}, avg. high loss: {:04f}".format(epoch, np.mean(losses), np.mean(lowlosses), np.mean(highlosses)))
        print("Epoch: {}, val. loss: {:04f}, val. low loss: {:04f}, val. high loss: {:04f}".format(epoch, np.mean(val_loss), np.mean(vallowloss), np.mean(valhighloss)))
        print()


if __name__ == '__main__':
    import argparse
    modes = ["generate", "learnfeatures", "learnmodel", "training_images_psnr", "val_images_psnr", "train_classifier", "learn_valfeatures", "train_segmentation"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="", help="Enter mode ({})".format("/".join(modes)), choices=modes)
    args = parser.parse_args()
    if args.mode == "generate":
        generate_data()
    elif args.mode == "learnfeatures":
        learn_features()
    elif args.mode == 'learn_valfeatures':
        learn_validation_features()
    elif args.mode == 'training_images_psnr':
        check_training_images()
    elif args.mode == 'val_images_psnr':
        check_training_images(train=False)
    elif args.mode == 'train_classifier':
        train_classifier()
    elif args.mode == "train_segmentation":
        train_segmentation()
    else:
        print("Invalid mode")
        exit(0)
