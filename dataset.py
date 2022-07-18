# modified from: https://github.com/POSTECH-CVLab/HighQualityFrameInterpolation
import torch.utils.data as data
from random import *
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random

try:
    import accimage
except ImportError:
    accimage = None



### Triplet test set ###
class SNU_FILM(Dataset):
    def  __init__(self, path='data/SNU-FILM/', mode='easy', transform=None):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        self.path = path
        test_root = os.path.join(path, 'test')
        test_fn = os.path.join(path, 'eval_modes/test-%s.txt' % mode)
        self.name = 'SNU_FILM'+mode
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]


        print("[%s] Test dataset has %d triplets" % (mode, len(self.frame_list)))
        self.transform = transform

    def path_corr(self, path):
        # print(path.split('data/SNU-FILM/'))
        path_last = path.split('data/SNU-FILM/')[1]
        return self.path + path_last


    def __getitem__(self, idx):
        imgpaths = self.frame_list[idx]

        img1 = Image.open(self.path_corr(imgpaths[0]))
        img2 = Image.open(self.path_corr(imgpaths[1]))
        img3 = Image.open(self.path_corr(imgpaths[2]))
        sample = {'frame1': img1, 'frame2': img2, 'frame3': img3}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.frame_list)

class x4k(data.Dataset):
    def __init__(self, path='../data/x4k/', transform=None):
        self.path = path
        self.img_list = []

        with open(path+'x4k.txt') as txt:
            seq_list = [line.strip() for line in txt]
        seq_list.sort()
        for seq in seq_list:
            t, f1, f2, f3 = seq.split('   ')
            t = float(t)
            f1 = path + 'test/' + f1
            f2 = path + 'test/' + f2
            f3 = path + 'test/' + f3
            self.img_list.append([f1, f2, f3, t])

        self.transform = transform
        self.name = 'x4k'

    def __getitem__(self, idx):

        fr_1 = Image.open(self.img_list[idx][0]).resize((2048, 1080), Image.ANTIALIAS)
        fr_2 = Image.open(self.img_list[idx][1]).resize((2048, 1080), Image.ANTIALIAS)
        fr_3 = Image.open(self.img_list[idx][2]).resize((2048, 1080), Image.ANTIALIAS)
        time = self.img_list[idx][3]

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 't': time, 'dir': self.img_list[idx][1], 'info': 0}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

class Xiph_4K_test(data.Dataset):
    def __init__(self, path='../data/Xiph/', t_frame=50, transform=None):
        self.path = path
        self.img_list = []
        for idx, dataset in enumerate(os.listdir(path+'test_4k')):
            # if idx != 13:
            #     print(idx, dataset)
            #     continue
            # print('*', idx, dataset)
            base_dir = path + 'test_4k/' + dataset

            for i in range(1, t_frame-2):
                im1_path = base_dir + '/' + '%03d' % (i + 0) + '.' + 'png'
                im2_path = base_dir + '/' + '%03d' % (i + 1) + '.' + 'png'
                im3_path = base_dir + '/' + '%03d' % (i + 2) + '.' + 'png'
                self.img_list.append([im1_path, im2_path, im3_path])

        self.transform = transform
        self.name = 'Xiph4K_full'

    def __getitem__(self, idx):

        fr_1 = Image.open(self.img_list[idx][0]).resize((2048, 1080), Image.ANTIALIAS)
        fr_2 = Image.open(self.img_list[idx][1]).resize((2048, 1080), Image.ANTIALIAS)
        fr_3 = Image.open(self.img_list[idx][2]).resize((2048, 1080), Image.ANTIALIAS)

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

class Vimeo90K_test_triplet(Dataset):
    def  __init__(self, path='data/vimeo_triplet/', transform=None):
        self.txt_path = path + 'tri_testlist.txt'
        self.path = path + 'sequences/'
        self.name = 'Vimeo90KTestTriplet'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1] + '/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()
        # Read image
        fr_1 = Image.open(septuplet + frames[0])
        fr_2 = Image.open(septuplet + frames[1])
        fr_3 = Image.open(septuplet + frames[2])
        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sep_list)

class nvidia_data_full(data.Dataset):
    def  __init__(self, path='data/nvidia_data_full/', frame=range(0, 22), transform=None):
        self.path = path
        self.img_list = []
        folder_name = '/dense/mv_images/'
        for dataset in os.listdir(self.path):
            if dataset == 'README.txt':
                continue
            for camera_idx in range(1, 13):
                camera_path = '/cam%02d.jpg' % (camera_idx)

                for i in frame:
                    im0 = self.path + dataset + folder_name + '%05d' % (i + 0) + camera_path
                    im1 = self.path + dataset + folder_name + '%05d' % (i + 1) + camera_path
                    im2 = self.path + dataset + folder_name + '%05d' % (i + 2) + camera_path
                    self.img_list.append([im0, im1, im2])

        self.transform = transform
        self.name = 'nvidia_data_full'


    def __getitem__(self, idx):

        fr_1 = Image.open(self.img_list[idx][0]).resize((540, 288), Image.ANTIALIAS)
        fr_2 = Image.open(self.img_list[idx][1]).resize((540, 288), Image.ANTIALIAS)
        fr_3 = Image.open(self.img_list[idx][2]).resize((540, 288), Image.ANTIALIAS)

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

class nvidia_data(Dataset):
    def  __init__(self, path='data//nvidia_data_full/', dataset='Balloon1-2', transform=None):
        self.path = path+dataset+'/dense/mv_images/'

        self.transform = transform
        self.img_list = []
        self.name = 'nvidia_data'
        self.dir_seq = os.listdir(self.path)
        self.dir_seq.sort()

    def __getitem__(self, idx):
        dataset_idx = int(idx / 12)
        camera_idx = int(idx % 12)

        camera_path = 'cam%02d.jpg'%(camera_idx + 1)
        # print(self.path + '/' + self.dir_seq[dataset_idx + 0] + '/' + camera_path)
        # print(self.path + '/' + self.dir_seq[dataset_idx + 1] + '/' + camera_path)
        # print(self.path + '/' + self.dir_seq[dataset_idx + 2] + '/' + camera_path)
        # exit()
        # Read image

        fr_1 = Image.open(self.path + '/' + self.dir_seq[dataset_idx+0] + '/' + camera_path).resize((540, 288), Image.ANTIALIAS)
        fr_2 = Image.open(self.path + '/' + self.dir_seq[dataset_idx+1] + '/' + camera_path).resize((540, 288), Image.ANTIALIAS)
        fr_3 = Image.open(self.path + '/' + self.dir_seq[dataset_idx+2] + '/' + camera_path).resize((540, 288), Image.ANTIALIAS)

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return (len(self.dir_seq)-2)*12

class nvidia_data_camera(Dataset):
    def  __init__(self, path='data//nvidia_data_full/', dataset='Balloon1-2',cam_i=1,dsize = (540, 288),transform=None):
        self.path = path+dataset+'/dense/mv_images/'

        self.transform = transform
        self.img_list = []
        self.name = 'nvidia_data'
        self.dir_seq = os.listdir(self.path)
        self.dir_seq.sort()
        self.cam_i = cam_i
        self.dsize = dsize



    def __getitem__(self, idx):
        dataset_idx = idx


        camera_path = 'cam%02d.jpg'%(self.cam_i + 1)
        # print(self.path + '/' + self.dir_seq[dataset_idx + 0] + '/' + camera_path)
        # print(self.path + '/' + self.dir_seq[dataset_idx + 1] + '/' + camera_path)
        # print(self.path + '/' + self.dir_seq[dataset_idx + 2] + '/' + camera_path)
        # exit()
        # Read image

        fr_1 = Image.open(self.path + '/' + self.dir_seq[dataset_idx+0] + '/' + camera_path).resize(self.dsize, Image.ANTIALIAS)
        fr_2 = Image.open(self.path + '/' + self.dir_seq[dataset_idx+1] + '/' + camera_path).resize(self.dsize, Image.ANTIALIAS)
        fr_3 = Image.open(self.path + '/' + self.dir_seq[dataset_idx+2] + '/' + camera_path).resize(self.dsize, Image.ANTIALIAS)

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return (len(self.dir_seq)-2)

### Triplet test set ###
class UCF101_test_triplet(Dataset):
    def __init__(self, path='data/ucf101/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []
        self.name = 'UCF101'
        self.dir_seq = os.listdir(self.path)
        self.dir_seq.sort()




    def __getitem__(self, idx):
        img_path = self.dir_seq[idx]

        # Read image
        fr_1 = Image.open(self.path+img_path+'/frame_00.png')
        fr_2 = Image.open(self.path+img_path+'/frame_01_gt.png')
        fr_3 = Image.open(self.path+img_path+'/frame_02.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.dir_seq)


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image or numpy.ndarray): Images to be converted to tensor.

        Returns:
            Tensor: Converted images.
        """
        if len(sample) == 5:
            fr_1, fr_2, fr_3, fr_4, fr_5 = sample['frame1'], sample['frame2'], sample['frame3'], sample['frame4'], sample[
                'frame5']
            pics = [fr_1, fr_2, fr_3, fr_4, fr_5]

            num = 0
            for pic in pics:
                if isinstance(pic, np.ndarray):
                    # handle numpy array
                    pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                    # backward compatibility
                    pic = pic.float().div(255)

                if accimage is not None and isinstance(pic, accimage.Image):
                    nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                    pic.copyto(nppic)
                    pic = torch.from_numpy(nppic)

                # handle PIL Image
                if pic.mode == 'I':
                    img = torch.from_numpy(np.array(pic, np.int32, copy=False))
                elif pic.mode == 'I;16':
                    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
                else:
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
                if pic.mode == 'YCbCr':
                    nchannel = 3
                elif pic.mode == 'I;16':
                    nchannel = 1
                else:
                    nchannel = len(pic.mode)
                img = img.view(pic.size[1], pic.size[0], nchannel)
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
                if isinstance(img, torch.ByteTensor):
                    pic = img.float().div(255)
                else:
                    pic = img

                pics[num] = pic
                num += 1

            return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2], 'frame4': pics[3], 'frame5': pics[4]}
        elif len(sample) == 3:
            fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']
            pics = [fr_1, fr_2, fr_3]

            num = 0
            for pic in pics:
                if isinstance(pic, np.ndarray):
                    # handle numpy array
                    pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                    # backward compatibility
                    pic = pic.float().div(255)

                if accimage is not None and isinstance(pic, accimage.Image):
                    nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                    pic.copyto(nppic)
                    pic = torch.from_numpy(nppic)

                # handle PIL Image
                if pic.mode == 'I':
                    img = torch.from_numpy(np.array(pic, np.int32, copy=False))
                elif pic.mode == 'I;16':
                    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
                else:
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
                if pic.mode == 'YCbCr':
                    nchannel = 3
                elif pic.mode == 'I;16':
                    nchannel = 1
                else:
                    nchannel = len(pic.mode)
                img = img.view(pic.size[1], pic.size[0], nchannel)
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
                if isinstance(img, torch.ByteTensor):
                    pic = img.float().div(255)
                else:
                    pic = img

                pics[num] = pic
                num += 1

            return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2]}
        elif len(sample) == 6:
            fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']
            t = sample['t']
            dir = sample['dir']
            info = sample['info']
            pics = [fr_1, fr_2, fr_3]

            num = 0
            for pic in pics:
                if isinstance(pic, np.ndarray):
                    # handle numpy array
                    pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                    # backward compatibility
                    pic = pic.float().div(255)

                if accimage is not None and isinstance(pic, accimage.Image):
                    nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                    pic.copyto(nppic)
                    pic = torch.from_numpy(nppic)

                # handle PIL Image
                if pic.mode == 'I':
                    img = torch.from_numpy(np.array(pic, np.int32, copy=False))
                elif pic.mode == 'I;16':
                    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
                else:
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
                if pic.mode == 'YCbCr':
                    nchannel = 3
                elif pic.mode == 'I;16':
                    nchannel = 1
                else:
                    nchannel = len(pic.mode)
                img = img.view(pic.size[1], pic.size[0], nchannel)
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
                if isinstance(img, torch.ByteTensor):
                    pic = img.float().div(255)
                else:
                    pic = img

                pics[num] = pic
                num += 1

            return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2], 't': t, 'dir': dir, 'info': info}
        elif len(sample) == 7:
            fr_1, fr_2, fr_3, fr_4, fr_5, fr_6, fr_7 = sample['frame1'], sample['frame2'], sample['frame3'], sample['frame4'], sample[
                'frame5'], sample['frame6'], sample['frame7']
            pics = [fr_1, fr_2, fr_3, fr_4, fr_5, fr_6, fr_7]

            num = 0
            for pic in pics:
                if isinstance(pic, np.ndarray):
                    # handle numpy array
                    pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                    # backward compatibility
                    pic = pic.float().div(255)

                if accimage is not None and isinstance(pic, accimage.Image):
                    nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                    pic.copyto(nppic)
                    pic = torch.from_numpy(nppic)

                # handle PIL Image
                if pic.mode == 'I':
                    img = torch.from_numpy(np.array(pic, np.int32, copy=False))
                elif pic.mode == 'I;16':
                    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
                else:
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
                if pic.mode == 'YCbCr':
                    nchannel = 3
                elif pic.mode == 'I;16':
                    nchannel = 1
                else:
                    nchannel = len(pic.mode)
                img = img.view(pic.size[1], pic.size[0], nchannel)
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
                if isinstance(img, torch.ByteTensor):
                    pic = img.float().div(255)
                else:
                    pic = img

                pics[num] = pic
                num += 1

            return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2], 'frame4': pics[3], 'frame5': pics[4], 'frame6': pics[5], 'frame7': pics[6]}

######################################################
# Transforms for triplet
class RandomHorizontalFlip_tri(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be flipped.

        Returns:
            PIL.Image: Randomly flipped images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']

        if random.random() < 0.5:
            fr_1 = fr_1.transpose(Image.FLIP_LEFT_RIGHT)
            fr_2 = fr_2.transpose(Image.FLIP_LEFT_RIGHT)
            fr_3 = fr_3.transpose(Image.FLIP_LEFT_RIGHT)

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}


class RandomVerticalFlip_tri(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be flipped.

        Returns:
            PIL.Image: Randomly flipped images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']

        if random.random() < 0.5:
            fr_1 = fr_1.transpose(Image.FLIP_TOP_BOTTOM)
            fr_2 = fr_2.transpose(Image.FLIP_TOP_BOTTOM)
            fr_3 = fr_3.transpose(Image.FLIP_TOP_BOTTOM)

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}


class ToTensor_tri(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image or numpy.ndarray): Images to be converted to tensor.

        Returns:
            Tensor: Converted images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']
        pics = [fr_1, fr_2, fr_3]

        num = 0
        for pic in pics:
            if isinstance(pic, np.ndarray):
                # handle numpy array
                pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                # backward compatibility
                pic = pic.float().div(255)

            if accimage is not None and isinstance(pic, accimage.Image):
                nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                pic.copyto(nppic)
                pic = torch.from_numpy(nppic)

            # handle PIL Image
            if pic.mode == 'I':
                img = torch.from_numpy(np.array(pic, np.int32, copy=False))
            elif pic.mode == 'I;16':
                img = torch.from_numpy(np.array(pic, np.int16, copy=False))
            else:
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            elif pic.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                pic = img.float().div(255)
            else:
                pic = img

            pics[num] = pic
            num += 1

        return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2]}


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be cropped.

        Returns:
            PIL.Image: Cropped images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']

        w, h = fr_1.size
        th = 256
        tw = 448

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        fr_1 = fr_1.crop((x1, y1, x1 + tw, y1 + th))
        fr_2 = fr_2.crop((x1, y1, x1 + tw, y1 + th))
        fr_3 = fr_3.crop((x1, y1, x1 + tw, y1 + th))

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

