import argparse

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.')

def DepthArgs(mode):
    if mode == 'test':
        parser.add_argument('--model_name', type=str, default='anti_densenet161_pad_nyu_256')
        parser.add_argument('--encoder', type=str, default='densenet161_bts')
        parser.add_argument('--dataset', type=str, default='azure_kinect_pad')
        parser.add_argument('--checkpoint_path', type=str, default='./estimator/models/depth_weight')
        parser.add_argument('--save_dir', type=str, default='./estimator/result')

        # image default ratio : 16:9 -> 32multiple -> 512*288
        parser.add_argument('--focal', type=float, default=252.1192)
        parser.add_argument('--input_width', type=int, default=512)
        parser.add_argument('--input_height', type=int, default=288)
        parser.add_argument('--max_depth', type=float, default=10)

        parser.add_argument('--do_kb_crop', action='store_true')
        parser.add_argument('--save_lpg', action='store_true')
        parser.add_argument('--bts_size', default=512)

        args = parser.parse_args()

        return args
