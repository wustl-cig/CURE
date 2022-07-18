from tqdm import tqdm
import argparse
from models.CURE import *
from dataset import *
import os

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps_times', '-f', default=2, type=int, help='fps X?')
    parser.add_argument('--video_dir', '-vdir', default='v1.mp4', type=str, help='dir of input video')
    parser.add_argument('--resolution', '-rs', default='1920,1080', type=str, help='resolution of output video')
    parser.add_argument('--working_dir', '-wdir', default='./video_data/', type=str, help='working dir for processing data')
    parser.add_argument('--eval_size', '-ebs', default=100000, type=int, help='eval batch size')
    parser.add_argument('--checkpoint_dir', '-dir', default='CURE', type=str, help='dir of checkpoint')
    parser.add_argument('--num_of_frames', '-nf', default=-1, type=int, help='numbers of frames to be interpolated')
    parser.add_argument('--down_sample_rate', '-d', default=1, type=int, help='downsample FPS? (must could be devided by 2)')
    return parser.parse_args()



def video_interpolate(args):
    assert args.down_sample_rate % 2 == 0 or args.down_sample_rate == 1 or args.down_sample_rate % 5 == 0
    int_round = int(math.log2(args.fps_times))
    idir, wdir = args.video_dir, args.working_dir
    odir = idir[:-4]+'X'+str(int_round)+'.mp4'
    odir_crop = idir[:-4] + '_Original' + '.mp4'
    if not os.path.exists(idir):
        print('Check your video path: '+idir)
        exit()
    else:
        print('Video: ' + idir)

    frame_dir_ori = wdir+os.path.splitext(idir)[0]+'frames/'
    frame_dir_itp = wdir+os.path.splitext(idir)[0]+'frames_interpolated/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    if not os.path.exists(frame_dir_ori):
        os.mkdir(frame_dir_ori)
    else:
        del_file(frame_dir_ori)

    if not os.path.exists(frame_dir_itp):
        os.mkdir(frame_dir_itp)
    else:
        del_file(frame_dir_itp)

    # read video to frames
    vidcap = cv2.VideoCapture(idir)
    if vidcap.isOpened():
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = vidcap.get(3)  # float
        height = vidcap.get(4)  # float
        print('Video: ', width, height, fps)
    else:
        print('video does not exsist')
        exit()
    vidcap = cv2.VideoCapture(idir)
    success, image = vidcap.read()
    count = 0
    while success:
        if args.resolution is not None:
            width, height = list(map(int, args.resolution.split(',')))
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if count % args.down_sample_rate == 0:
            cv2.imwrite(frame_dir_ori + '00_frame%09d.png' % count, image)
        success, image = vidcap.read()
        count += 1
        if args.num_of_frames != -1:
            if count == args.num_of_frames * args.down_sample_rate:
                break
    # initialize network
    model = CURE()
    if '.pth.tar' in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    elif '.pth.tar' not in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir + '.pth.tar'

    data = torch.load(checkpoint_dir)
    if 'state_dict' in data.keys():
        model.load_state_dict(data['state_dict'])
    else:
        model.load_state_dict(data)
    model.cuda()
    model.eval()

    frame_time = 0.5
    print('Total ' + str(int_round) + ' round')

    # frameList = os.listdir(frame_dir_ori)
    frameList = []
    for f in os.listdir(frame_dir_ori):
        if f.endswith('.png'):
            frameList.append(f)
    frameList.sort()
    print('Number of frames: %d' % len(frameList))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writeDir = odir_crop
    videoWriter = cv2.VideoWriter(writeDir, fourcc, fps/args.down_sample_rate, (int(width), int(height)), True)
    videoWriter.release()
    for frame_dir in frameList:
        f_dir = frame_dir_ori + '/' + frame_dir
        frame = cv2.imread(f_dir)
        videoWriter.write(frame)

    frameList = os.listdir(frame_dir_ori)
    # compute total frames
    total_frame_cnt = 0
    init_frame_len = len(frameList)
    for _ in range(int_round):
        total_frame_cnt = total_frame_cnt + init_frame_len - 1
        init_frame_len = init_frame_len * 2 - 1

    with tqdm(total=total_frame_cnt) as t:
        for rnd in range(int_round):
            frameList = os.listdir(frame_dir_ori)
            frameList.sort()
            for f in range(len(frameList[:-1])):
                # print('Frame: '+frameList[f])

                f1 = cv2.imread(frame_dir_ori + frameList[f])
                f2 = cv2.imread(frame_dir_ori + frameList[f + 1])
                im1, im2 = imProcess(args, f1, f2)

                f12, _ = pred_frame(args, im1, im2, model, None, time=frame_time)
                f12 = frame_rec(f12)
                cv2.imwrite(frame_dir_itp + frameList[f][:-4] + '_0.png', f1)
                cv2.imwrite(frame_dir_itp + frameList[f][:-4] + '_' + str(2 ** (0)) + '.png', f12)

                del im1, im2, f12
                torch.cuda.empty_cache()
                t.set_postfix(Fm=frameList[f][-10:], Rd=rnd+1)
                t.update(1)

            f = cv2.imread(frame_dir_ori + frameList[-1])
            cv2.imwrite(frame_dir_itp + frameList[-1][:-4] + '_0.png', f)
            shutil.rmtree(frame_dir_ori)
            os.rename(frame_dir_itp, frame_dir_ori)
            if not os.path.exists(frame_dir_itp):
                os.mkdir(frame_dir_itp)

    print('Complete!!!')
    print('Writting video')

    frameList = []
    for f in os.listdir(frame_dir_ori):
        if f.endswith('.png'):
            frameList.append(f)
    frameList.sort()
    print('Number of frames: %d' % len(frameList))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writeDir = odir
    videoWriter = cv2.VideoWriter(writeDir, fourcc, fps*(2**int_round)/args.down_sample_rate, (int(width), int(height)), True)
    for frame_dir in frameList:
        f_dir = frame_dir_ori + '/' + frame_dir
        frame = cv2.imread(f_dir)
        videoWriter.write(frame)
    videoWriter.release()
    del_file(frame_dir_itp)
    del_file(frame_dir_ori)

if __name__ == '__main__':
    args = args_parser()
    print(args)
    with torch.no_grad():
        video_interpolate(args)