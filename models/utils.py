import math
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import interpolate

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).cuda()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window

def compare_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def compare_psnr(pred, gt):

    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def showFlow(flow_viz, flo, path='./results/flo.png'):
    # plot a figure of optical flow
    flo = flo[0].permute(1, 2, 0).cpu().detach().numpy()
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(path, np.array(flo[:, :, [2, 1, 0]], np.uint8))


def im_rec(im):
    im = (im * 0.5 + 0.5)[0].permute(1, 2, 0).cpu().numpy() * 255
    im = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    return np.array(im, dtype=np.uint8)

class batch_converter():
    def __init__(self, bs, q):
        self.bs = bs
        self.q = q

    def __fit__(self, x):
        x_dim = list(x.shape)
        x_dim.pop(0)
        x_dim[0] = self.bs * self.q
        return x.view(x_dim)

    def __unfit__(self, x):
        x_dim = list(x.shape)
        x_dim[0] = self.q
        x_dim.insert(0, self.bs)
        return x.view(x_dim)

    def fit(self, *inputs):
        return [self.__fit__(x) for x in inputs]

    def unfit(self, *inputs):
        return [self.__unfit__(x) for x in inputs]


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def coordinate_grid(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def generate_coorMap(shape, scale=True, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    if scale:
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
    else:
        for i, n in enumerate(shape):
            seq = torch.arange(n)
            coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def generate_flow_time(flow12, flow21, time_stamp):
    # flow1t: frame1 -> frame t
    flow1t = flow12 * time_stamp
    # flow2t: frame2 -> frame t
    flow2t = flow21 * (1.0 - time_stamp)
    return flow1t, flow2t

def generate_double_flow_time(flow12, flow21, time_stamp):
    # flow1t: frame1 -> frame t
    flowb1t = flow12 * time_stamp
    # flow2t: frame2 -> frame t
    flowf2t = flow21 * (1.0 - time_stamp)

    flowf1t = - time_stamp * flow21

    flowb2t = - (1.0 - time_stamp) * flow12
    return flowf1t, flowf2t, flowb1t, flowb2t

def generate_double_coormap(flowf1t, flowf2t, flowb1t, flowb2t, coorMap):
    coorMap = coorMap.unsqueeze(0).repeat_interleave(flowf1t.shape[0], dim=0)
    coorMap = torch.cat((coorMap[:, :, :, 1:2], coorMap[:, :, :, 0:1]), dim=-1)
    # img1 -> imgt
    coorMapf1_t = coorMap.permute(0, 3, 1, 2) + flowf1t
    coorMapf1_t = coorMapf1_t.permute(0, 3, 2, 1)
    coorMapb1_t = coorMap.permute(0, 3, 1, 2) + flowb1t
    coorMapb1_t = coorMapb1_t.permute(0, 3, 2, 1)
    # img2 -> imgt
    coorMapf2_t = coorMap.permute(0, 3, 1, 2) + flowf2t
    coorMapf2_t = coorMapf2_t.permute(0, 3, 2, 1)
    coorMapb2_t = coorMap.permute(0, 3, 1, 2) + flowb2t
    coorMapb2_t = coorMapb2_t.permute(0, 3, 2, 1)
    return coorMapf1_t, coorMapb1_t, coorMapf2_t, coorMapb2_t


def generate_coormap(flow1t, flow2t, coorMap):
    coorMap = coorMap.unsqueeze(0).repeat_interleave(flow1t.shape[0], dim=0)
    coorMap = torch.cat((coorMap[:, :, :, 1:2], coorMap[:, :, :, 0:1]), dim=-1)
    # img1 -> imgt
    coorMap1_t = coorMap.permute(0, 3, 1, 2) + flow1t
    coorMap1_t = coorMap1_t.permute(0, 3, 2, 1)
    # img2 -> imgt
    coorMap2_t = coorMap.permute(0, 3, 1, 2) + flow2t
    coorMap2_t = coorMap2_t.permute(0, 3, 2, 1)
    return coorMap1_t, coorMap2_t

def generate_coormapntime(flow12, flow21, coorMap, timeStamp=0.5):
    # flow1t: frame1 -> frame t
    flow1t = flow12 * timeStamp
    # flow2t: frame2 -> frame t
    flow2t = flow21 * (1.0 - timeStamp)
    coorMap = torch.cat((coorMap[:, :, 1:2], coorMap[:, :, 0:1]), dim=-1)
    # img1 -> imgt
    coorMap1_t = coorMap.permute(2, 0, 1).unsqueeze(0) + flow1t
    coorMap1_t = coorMap1_t[0].permute(2, 1, 0)
    # img2 -> imgt
    coorMap2_t = coorMap.permute(2, 0, 1).unsqueeze(0) + flow2t
    coorMap2_t = coorMap2_t[0].permute(2, 1, 0)

    return coorMap1_t, coorMap2_t

def coorEncodeMapSingle(coor, num_encoding_fn_xy=10):
    # output [width, height, data]
    # coor = np.array([[[x, y]]])
    coor = coor.unsqueeze(0).cpu().numpy()
    coorTb = None

    for l in range(num_encoding_fn_xy):
        if coorTb is None:
            coorTb = np.sin(np.dot((2 ** l) * math.pi, coor/1.1), dtype=np.double)
            coorTb = np.dstack((coorTb, np.cos(np.dot((2 ** l) * math.pi, coor/1.1), dtype=np.double)))
        else:
            coorTb = np.dstack((coorTb, np.sin(np.dot((2 ** l) * math.pi, coor/1.1), dtype=np.double)))
            coorTb = np.dstack((coorTb, np.cos(np.dot((2 ** l) * math.pi, coor/1.1), dtype=np.double)))
    return coorTb[0, :, :]

def timeEncode(time, num_encoding_fn_time=4):
# encode time to high dimention
# input: time
# output: [time * 2 * num_encoding_fn_time]
    timeTb = None
    for l in range(num_encoding_fn_time):
        if timeTb is None:
            timeTb = np.sin(np.dot((2 ** l) * math.pi, time/1.1))
            timeTb = np.append(timeTb, np.cos(np.dot((2 ** l) * math.pi, time/1.1), dtype=np.double))
        else:
            timeTb = np.append(timeTb, np.sin(np.dot((2 ** l) * math.pi, time/1.1), dtype=np.double))
            timeTb = np.append(timeTb, np.cos(np.dot((2 ** l) * math.pi, time/1.1), dtype=np.double))
    return timeTb


import os
import shutil

def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def frame_rec(im):
    im = (im)[0].permute(1, 2, 0).cpu().numpy() * 255
    im = cv2.cvtColor(np.array(im, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    return np.array(im, dtype=np.uint8)


def imProcess(args, im1, im2):
    imNorm = lambda x: (x - 0.5) * 2
    # im1 = torch.FloatTensor(numpy.ascontiguousarray(
    #     im1[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).unsqueeze(0)
    # im2 = torch.FloatTensor(numpy.ascontiguousarray(
    #     im2[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).unsqueeze(0)
    im1 = torch.FloatTensor(im1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)).unsqueeze(0)
    im2 = torch.FloatTensor(im2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)).unsqueeze(0)

    im1 = im1.cuda()
    im2 = im2.cuda()
    return imNorm(im1), imNorm(im2)

def batched_predict(model, im1, im2, coor, bsize=900000, time=0.5):
    # predict a batch of pixels
    with torch.no_grad():
        n = coor.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model(coor[:, ql: qr, :], im1, im2, time_stamp=time)[0]
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=0)
    return pred

def pred_frame(args, im1, im2, model, coorMap = None, time=0.5):
    height, width = im1.shape[-2:]
    if coorMap is None:
        coorMap = generate_coorMap((height, width), scale=False).unsqueeze(0).cuda()
    model.flow_estimate(im1, im2)
    fe1, fe2 = model.generate_feature(im1, im2)
    model.warp(fe1, fe2, time=time)
    pred = batched_predict(model, fe1, fe2, coorMap, bsize=args.eval_size, time=time)
    h,w = im1.shape[2:4]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
    del im1, im2
    del fe1, fe2
    del model.flow21, model.flow12
    del model.fe_concat
    torch.cuda.empty_cache()
    return pred, coorMap

