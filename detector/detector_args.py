import argparse


def DetectorArgs(mode=None):
    '''
    Detector Package 전용 Arguments
    type : mask(segment)와 point(landmark) 설정만 유의할 것.
    '''
    parser = argparse.ArgumentParser(description="Segment & Landmark implementation.")
    parser.add_argument('--type', type=str, choices=['mask', 'point'], default='point',
                        help='Dataset type is model type')
    parser.add_argument('--key_type', type=str, choices=['pad', 'bra'], default='bra',
                        help='Dataset type is model type')
    parser.add_argument('--data_root', type=str, default='/home/ljj/dataset')
    parser.add_argument('--dataset', type=str, default='train')

    if mode == 'dataset':
        parser.add_argument('--ratio', type=float, default=[0.9, 0.1, 0.], nargs=3, metavar=('train', 'val', 'test'))
        parser.add_argument('--seed', type=int, default=10)

    elif mode == 'train':
        # System Settings
        parser.add_argument('--gpu_id', type=int, default=0)

        # OPTIMIZER
        parser.add_argument('--opt', type=str, default='SGD')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--lr_schedule', type=str, default='Step')
        parser.add_argument('--lr_step', type=int, default=5)
        parser.add_argument('--weight_decay', type=int, default=0.00001)
        parser.add_argument('--momentum', type=float, default=0.9)

        # Runtime
        parser.add_argument('--epochs', type=int, default=51)
        parser.add_argument('--save_interval', type=int, default=5)
        parser.add_argument('--print_freq', type=int, default=4)

        # DATA
        parser.add_argument('--batch-size', type=int, default=1)
        parser.add_argument('--workers', type=int, default=8)
 

    elif mode == 'test':
        parser.add_argument('--mask_model', type=str, default='/home/ljj/models/mask_weight_0216.pth')
        parser.add_argument('--point_model', type=str, default='/home/ljj/models/point_weight_0216.pth')
        parser.add_argument('--u2n_model', type=str, default='/home/ljj/models/u2net_io_0216.pth')
        parser.add_argument('--test_folder', type=str, default='/home/ljj/dataset/PAD/102samples/images')

    args = parser.parse_args(args=[])
    return args
