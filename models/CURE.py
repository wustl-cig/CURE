import torch.nn.functional as F
import pytorch_lightning as pl
from .FENet import *
from .raftcore.raft import *
from .utils import *


imNorm = lambda x: (x - 0.5) * 2


class Encoder(nn.Module):
    def __init__(self, in_nc, out_nc, stride, k_size, pad):
        super(Encoder, self).__init__()

        self.padd = nn.ReplicationPad2d(pad)
        self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        s = self.padd(x)
        s = self.Conv(s)
        s = self.relu(s)
        return s

class Decoder(nn.Module):
    def __init__(self, in_nc, out_nc, stride, k_size, pad):
        super(Decoder, self).__init__()

        self.padd = nn.ReplicationPad2d(pad)
        self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x, x_c):
        x = F.interpolate(x, size=x_c.shape[2: 4], mode='bilinear', align_corners=False).type_as(x)
        x = torch.cat([x, x_c], 1).type_as(x)
        s = self.padd(x)
        s = self.Conv(s)
        s = self.relu(s)
        return s

class Fuser(pl.LightningModule):
    def __init__(self, n_feats=64, freeze=False):
        super(Fuser, self).__init__()
        self.enc1 = Encoder(n_feats * 2, n_feats * 2, stride=2, k_size=3, pad=1)
        self.enc2 = Encoder(n_feats * 2, n_feats * 4, stride=2, k_size=3, pad=1)
        self.enc3 = Encoder(n_feats * 4, n_feats * 8, stride=2, k_size=3, pad=1)
        self.enc4 = Encoder(n_feats * 8, n_feats * 8, stride=2, k_size=3, pad=1)

        self.dec1 = Decoder(n_feats * 8 + n_feats * 8, n_feats * 8, stride=1, k_size=3, pad=1)
        self.dec2 = Decoder(n_feats * 8 + n_feats * 4, n_feats * 4, stride=1, k_size=3, pad=1)
        self.dec3 = Decoder(n_feats * 4 + n_feats * 2, n_feats * 2, stride=1, k_size=3, pad=1)
        self.dec4 = Decoder(n_feats * 2 + n_feats * 2, n_feats, stride=1, k_size=3, pad=1)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, *inputs):
        x = torch.cat(inputs, 1)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        d1 = self.dec1(s4, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)
        out = self.dec4(d3, x)
        return out


class INR(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_feats, pos_ecding=[1, 1], freeze=False):
        super(INR, self).__init__()
        self.n_feats = n_feats
        self.layers = torch.nn.ModuleList()
        lastv = in_dim
        self.in_dim = in_dim
        self.hidden_list = [256, 256 + n_feats, 256, 256 + n_feats, 256]
        for idx, hidden  in enumerate(self.hidden_list):
            self.layers.append(nn.Linear(lastv, 256))
            self.layers.append(nn.ReLU())
            lastv = hidden
        self.layers.append(nn.Linear(lastv, out_dim))
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, feat, coor=None, time=None):
        x = feat
        shape = x.shape[:-1]
        feat = feat.reshape(-1, feat.shape[-1]).float()
        x = x.reshape(-1, x.shape[-1]).float()
        coor = coor.reshape(-1, coor.shape[-1]).float()
        time = time.reshape(-1, time.shape[-1]).float()

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):
                x = layer(x)
            elif idx == 0:
                x = layer(torch.cat((coor, time, x), -1))
            elif layer.in_features == 256 + self.n_feats:
                x = layer(torch.cat((feat, x), -1))
            else:
                x = layer(x)
        return x.view(*shape, -1)


class CURE(pl.LightningModule):
    def __init__(self, learning_rate=0.00001, batch_size=1, cell=1, n_feats=64):
        super().__init__()
        self.n_feats = n_feats
        self.height, self.width = 0, 0
        self.coorMap = None
        self.cell = cell * 2 + 1
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # model architecture
        self.FE = make_fenet(n_feats=n_feats)
        self.OF = raft_ready()
        self.Fuser = Fuser(n_feats=n_feats*2)
        self.fe_concat = None
        mlp_feats =  n_feats * self.cell * self.cell * 2
        mlp_din = mlp_feats + 3

        self.INR = INR(in_dim=mlp_din, out_dim=3, n_feats=mlp_feats)

    def generate_feature(self, *inputs):
        return [self.FE(x) for x in inputs]

    def flow_estimate(self, image1, image2, scale=True, test_mode=True):
        self.height, self.width = image1.shape[-2:]
        self.coorMap = generate_coorMap((self.height, self.width), scale=scale, flatten=False).type_as(image1)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # flow12: frame2 -> frame1
        _, self.flow21 = self.OF(image1, image2, iters=50, test_mode=test_mode)
        # flow12: frame1 -> frame2
        _, self.flow12 = self.OF(image2, image1, iters=50, test_mode=test_mode)
        self.flow21, self.flow12 = padder.unpad(self.flow21), padder.unpad(self.flow12)
        if scale:
            self.flow12 = torch.cat([self.flow12[:, 0: 1, :, :] / ((self.flow12.shape[3] - 1.0) / 2.0),
                                     self.flow12[:, 1: 2, :, :] / ((self.flow12.shape[2] - 1.0) / 2.0)], 1)
            self.flow21 = torch.cat([self.flow21[:, 0: 1, :, :] / ((self.flow21.shape[3] - 1.0) / 2.0),
                                     self.flow21[:, 1: 2, :, :] / ((self.flow21.shape[2] - 1.0) / 2.0)], 1)
        return self.flow21, self.flow12

    def warp(self, im1, im2, time=0.5):

        fe1, fe2 = im1, im2
        flowf1t, flowf2t, flowb1t, flowb2t = generate_double_flow_time(self.flow12, self.flow21, time_stamp=time)
        flowf1t, flowf2t, flowb1t, flowb2t = flowf1t.type_as(im1), flowf2t.type_as(im1), flowb1t.type_as(im1), flowb2t.type_as(im1)
        coorMapf1_t, coorMapb1_t, coorMapf2_t, coorMapb2_t = generate_double_coormap(flowf1t, flowf2t, flowb1t, flowb2t, self.coorMap)
        fef1 = nn.functional.grid_sample(fe1, coorMapf1_t, mode='bilinear', padding_mode='reflection')
        feb1 = nn.functional.grid_sample(fe1, coorMapb1_t, mode='bilinear', padding_mode='reflection')
        fef2 = nn.functional.grid_sample(fe2, coorMapf2_t, mode='bilinear', padding_mode='reflection')
        feb2 = nn.functional.grid_sample(fe2, coorMapb2_t, mode='bilinear', padding_mode='reflection')
        pad_stride = int((self.cell - 1) / 2)
        pad = nn.ReflectionPad2d(padding=(pad_stride, pad_stride, pad_stride, pad_stride))
        fef1, feb1, fef2, feb2 = pad(fef1), pad(feb1), pad(fef2), pad(feb2)
        fe_concat = self.Fuser(fef1, feb1, fef2, feb2)
        self.fe_concat = fe_concat
        return fe_concat

    def forward(self, coor, im1, im2, time_stamp=0.5):
        fullfeaturemap = self.fe_concat
        bs, q = coor.shape[0: 2]
        h, w = im1.shape[-2], im1.shape[-1]
        coor_in = torch.cat([coor[:, :, 0: 1] / ((h - 1.0) / 2.0), coor[:, :, 1: 2] / ((w - 1.0) / 2.0)], -1) - 1
        pad_stride = int((self.cell - 1) / 2)
        full_feature_map_shape = [fullfeaturemap.shape[0], fullfeaturemap.shape[1] * 9, fullfeaturemap.shape[2]-pad_stride*2, fullfeaturemap.shape[3]-pad_stride*2]
        fullfeaturemap = F.unfold(fullfeaturemap, self.cell, padding=0).view(full_feature_map_shape[0], full_feature_map_shape[1], full_feature_map_shape[2], full_feature_map_shape[3])
        feature_vector = F.grid_sample(fullfeaturemap, coor_in.unsqueeze(1), mode='nearest', align_corners=False, padding_mode='reflection')[:, :, 0, :].permute(0, 2, 1)
        time_stamp_tensor = torch.tensor(time_stamp).type_as(fullfeaturemap)
        time_stamp_tensor = time_stamp_tensor.repeat(1, q, 1).permute(2, 1, 0)
        I = self.INR(feature_vector, coor_in, time_stamp_tensor)
        return I
