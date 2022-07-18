import argparse
from torch.utils.data import DataLoader, random_split
from models.CURE import *
from dataset import *
import time
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_worker', '-nw', default=5, type=int, help='number of workers to load data by dataloader')
    parser.add_argument('--eval_size', '-ebs', default=200000, type=int, help='eval batch size')
    parser.add_argument('--checkpoint_dir', '-dir', default='CURE', type=str,
                        help='dir or checkpoint')
    parser.add_argument('--dataset', '-d', default='ucf101', type=str,help='dataset',
                        choices={"ucf101", "vimeo90k", "sfeasy", "sfmedium", "sfhard", "sfextreme", "nvidia", "xiph4k", "x4k"})
    parser.add_argument('--dataset_basedir', '-dbdir', default='./', type=str,
                        help='dir or checkpoint')
    return parser.parse_args()

def test():
    args = args_parser()
    model = CURE(batch_size=args.eval_size)
    if '.pth.tar' in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    elif '.pth.tar' not in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir+'.pth.tar'

    data = torch.load(checkpoint_dir)
    if 'state_dict' in data.keys():
       model.load_state_dict(data['state_dict'])
    else:
       model.load_state_dict(data)
    model.cuda()
    model.eval()
    transform_to_tensors = transforms.Compose([ToTensor()])
    if args.dataset == "ucf101":
        test_dataset = UCF101_test_triplet(path=args.dataset_basedir + 'data/ucf101/', transform=transform_to_tensors)
    elif args.dataset == "vimeo90k":
        test_dataset = Vimeo90K_test_triplet(path=args.dataset_basedir + 'data/vimeo_triplet/', transform=transform_to_tensors)
    elif args.dataset == "nvidia":
        test_dataset = nvidia_data_full(path=args.dataset_basedir + 'data/nvidia_data_full/', transform=transform_to_tensors)
    elif args.dataset == "xiph4k":
        test_dataset = Xiph_4K_test(path=args.dataset_basedir + 'data/Xiph/', t_frame=50, transform=transform_to_tensors)
    elif args.dataset == "snufilm":
        test_dataset = Vimeo90K_test_triplet(path=args.dataset_basedir + 'data/vimeo_triplet/', transform=transform_to_tensors)
    elif args.dataset == "sfeasy":
        test_dataset = SNU_FILM(path=args.dataset_basedir + 'data/SNU-FILM/', mode='easy', transform=transform_to_tensors)
    elif args.dataset == "sfmedium":
        test_dataset = SNU_FILM(path=args.dataset_basedir + 'data/SNU-FILM/', mode='medium', transform=transform_to_tensors)
    elif args.dataset == "sfhard":
        test_dataset = SNU_FILM(path=args.dataset_basedir + 'data/SNU-FILM/', mode='hard', transform=transform_to_tensors)
    elif args.dataset == "sfextreme":
        test_dataset = SNU_FILM(path=args.dataset_basedir + 'data/SNU-FILM/', mode='extreme', transform=transform_to_tensors)
    elif args.dataset == 'x4k':
        test_dataset = x4k(path=args.dataset_basedir + 'data/x4k/', transform=transform_to_tensors)
    else:
        exit("Not impletement")

    dataset_name = test_dataset.name
    print('\n' + dataset_name)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True
    )
    imNorm = lambda x: (x - 0.5) * 2
    loss, psnr, ssim = [], [], []
    with tqdm(total=len(test_loader)) as t:
        for iter, batch in enumerate(test_loader):
            torch.cuda.empty_cache()
            im1 = batch['frame1'].cuda()
            gt = batch['frame2'].cuda()
            im2 = batch['frame3'].cuda()
            try:
                time_frame = batch['t'].item()
            except:
                time_frame = 0.5
            im1 = imNorm(im1)
            im2 = imNorm(im2)
            pred, _ = pred_frame(args, im1, im2, model, None, time=time_frame)
            psnr.append(compare_psnr(pred, gt))
            ssim.append(compare_ssim(pred, gt).cpu())
            del im1, gt, im2, pred
            torch.cuda.empty_cache()
            t.set_postfix(psnr=round(sum(psnr) / len(psnr), 4), ssim=(sum(ssim) / len(ssim)).cpu().detach().numpy())
            t.update(1)
    current_time = time.asctime()
    psnr_avg = sum(psnr) / len(psnr)
    ssim_avg = sum(ssim) / len(ssim)
    repo_dir = './test_report.txt'
    repo = open(repo_dir, 'a')
    print(' ', file=repo)
    print(' ', file=repo)
    print(' ', file=repo)
    print('==================', file=repo)
    print('******************', file=repo)
    print('==================', file=repo)
    print(current_time, file=repo)
    print('dataset: ', dataset_name, file=repo)
    print('Ours', file=repo)
    print('PSNR: ' + str(psnr_avg), file=repo)
    print('SSIM: %.6f' % ssim_avg, file=repo)
    print('PSNR: ' + str(psnr_avg))
    print('SSIM: ' + str(ssim_avg))
    repo.close()
    np.save('Ours' + dataset_name + 'psnr.npy', psnr)
    np.save('Ours' + dataset_name + 'ssim.npy', ssim)



if __name__ == '__main__':
    args = args_parser()
    print(args)
    with torch.no_grad():
        test()
