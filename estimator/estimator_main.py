import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from common import timefn2
from .anti_args import DepthArgs
from .anti_dataloader import AntiDataLoader
from .bts import BtsModel


class AntiDepths(object):
    def __init__(self):
        self.args = DepthArgs(mode='test')

    @timefn2
    def predict(self, images, filename, DEBUG=False):

        self.args.mode = 'test'
        dataloader = AntiDataLoader(self.args, images, 'test')

        model = BtsModel(params=self.args)
        model = torch.nn.DataParallel(model)

        checkpoint = torch.load(self.args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.cuda()

        num_test_samples = len(images)

        pred_depths = []
        result = []
        with torch.no_grad():
            for _, sample in enumerate(tqdm(dataloader.data)):
                image = Variable(sample['image'].cuda())
                focal = Variable(sample['focal'].cuda())
                # Predict
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
                pred_depths.append(depth_est.cpu().numpy().squeeze())

        for s in tqdm(range(num_test_samples)):

            pred_depth = pred_depths[s]
            pred_depth_scaled = pred_depth * 256.0

            pred_depth_scaled = pred_depth_scaled.astype(np.uint8)

            if DEBUG:
                if not self.args.save_dir:
                    os.mkdir(self.args.save_dir)

                filename_pred_png = '{}/{}'.format(self.args.save_dir, filename[s].split('\\')[-1])
                cv2.imwrite(filename_pred_png, pred_depth_scaled)

            result.append(pred_depth_scaled)

        return result
