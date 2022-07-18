import argparse
from models.CURE import *
from dataset import *


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', '-ebs', default=200000, type=int, help='eval batch size')
    parser.add_argument('--checkpoint_dir', '-dir', default='CURE', type=str, help='dir or checkpoint')
    parser.add_argument('--frame0', '-f0', default='test_input/frame_0021.png', type=str, help='dir of frame 0')
    parser.add_argument('--frame1', '-f1', default='test_input/frame_0022.png', type=str, help='dir of frame 1')
    parser.add_argument('--frame_out', '-fo', default='predict.png', type=str, help='dir of output frame')
    parser.add_argument('--time_frame', '-t', default=0.5, type=float, help='time of interpolated frame [0, 1]')
    return parser.parse_args()

def test():
    args = args_parser()
    if not os.path.exists(args.frame0) or not os.path.exists(args.frame1):
        exit('frames does not found')
    model = CURE(batch_size=args.eval_size)
    checkpoint_dir = ''
    if '.pth.tar' in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    elif '.pth.tar' not in args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir + '.pth.tar'
    ckpt_data = torch.load(checkpoint_dir)
    if 'state_dict' in ckpt_data.keys():
       model.load_state_dict(ckpt_data['state_dict'])
    else:
       model.load_state_dict(ckpt_data)
    model.cuda()
    model.eval()
    im0_np = cv2.imread(args.frame0)
    im2_np = cv2.imread(args.frame1)
    im1, im2 = imProcess(args, im0_np, im2_np)
    pred, _ = pred_frame(args, im1, im2, model, None, time=args.time_frame)
    pred = frame_rec(pred)
    cv2.imwrite(args.frame_out, pred)
    print("Complete: ", args.frame_out)

if __name__ == '__main__':
    args = args_parser()
    print(args)
    with torch.no_grad():
        test()
