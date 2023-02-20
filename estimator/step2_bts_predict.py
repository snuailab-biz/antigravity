# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import glob
import argparse
import time

from torch.autograd import Variable
from bts import BtsModel

from tqdm import tqdm
from bts_dataloader import *

from common import timefn2


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def depth_args():
    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.')

    parser.add_argument('--model_name',         type=str,   default='anti_densenet161_pad_nyu_256')
    parser.add_argument('--encoder',            type=str,   default='densenet161_bts')
    parser.add_argument('--dataset',            type=str,   default='azure_kinect_pad')
    parser.add_argument('--checkpoint_path',    type=str,   default='./models/anti_densenet161_pad_nyu_256/model-10000')
    parser.add_argument('--data_path',          type=str,   default='./dataset/dataset_test_iphone')
    parser.add_argument('--filenames_file',     type=str,   default='./dataset/dataset_test_iphone/test_file_list.txt')

    # image default ratio : 16:9 -> 32multiple -> 512*288
    parser.add_argument('--focal',              type=float, default=252.1192)
    parser.add_argument('--input_width',        type=int,   default=512)
    parser.add_argument('--input_height',       type=int,   default=288)
    parser.add_argument('--max_depth',          type=float, default=10)

    parser.add_argument('--do_kb_crop',         action='store_true')
    parser.add_argument('--save_lpg',           action='store_true')
    parser.add_argument('--bts_size',           default=512)

    args = parser.parse_args()

    return args


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


@timefn2
def predict(args):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))

    if not args.data_path.find('iphone') == -1:
        save_name = '{}/result_{}'.format(args.data_path, args.model_name)
    else:
        save_name = './dataset/result_{}'.format(args.model_name)

    print('Saving result pngs..')
    if not os.path.exists(save_name):
        os.mkdir(save_name)

    for s in tqdm(range(num_test_samples)):

        filename_pred_png = '{}/{}'.format(save_name, lines[s].split()[0].split('/')[-1])
        pred_depth = pred_depths[s]
        #data-type
        if args.model_name == 'nyu':
            pred_depth_scaled = pred_depth * 1000.0

        elif args.model_name == 'azure_kinect_pad':
            pred_depth_scaled = pred_depth * float(args.model_name.split('_')[-1])

        else:
            pred_depth_scaled = pred_depth * 256.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint8)
        cv2.imwrite(filename_pred_png, pred_depth_scaled)

    return

if __name__ == '__main__':

    args = depth_args()
    predict(args)
