import common
from common.common import timefn2
from volume_estimate.volume_estimate_mainGT import VolumeEstimateGT
from detector.detector_main import AntiDetector

import numpy as np

import pandas as pd
import csv


class FittingRoomAI_GT(object):

    def __init__(self, DEBUG, save_path):
        self.logger = common.get_runtime_logger()
        self.save_path=save_path
        self.detector = AntiDetector(DEBUG=DEBUG,save_path=self.save_path)
        self.volumes = VolumeEstimateGT(DEBUG)

    @timefn2
    def detectorProcess(self, images, filename):
        '''
        pred_mask : mask부분이 색칠된 image
        pred_keypoint : Keypoints에 대해 색칠된 image
        pred_info : {'bbox': {'left': bbox, 'right': bbox}, 'landmark': {'left':Keypoints, 'right': keypoints}
        '''
        pred_mask, pred_keypoint, pred_info = self.detector.predict(images, filename)

        self.logger.info("mask segmentation complete")
        return pred_mask, pred_info


    @timefn2
    def volumeProcess(self, pred_depth, pred_mask, pred_info):
        # keypoints (Left, Bottom, Right)를 이용하여 afm 정의
        # run : output volume
        pad_pcd_l, landmark_points_l, affine_l, interpo_depth_pcd_l = self.volumes.volume_preprocessing(pred_depth, pred_mask, pred_info['landmark'], locate='left')
        pad_pcd_r, landmark_points_r, affine_r, interpo_depth_pcd_r = self.volumes.volume_preprocessing(pred_depth, pred_mask, pred_info['landmark'], locate='right')

        if not np.isnan(affine_l).any(): # Landmark가 잘못 찍힌 경우 affine transform을 구하지 못하는 경우가 있어 실패
            left_volume_pad, left_volume = self.volumes.volume_estimate(pad_pcd_l, landmark_points_l, affine_l, interpo_depth_pcd_l)
        else:
            left_volume_pad, left_volume = 0, 0 # Landmark가 잘못 찍힌 경우 affine transform을 구하지 못하는 경우가 있어 실패

        if not np.isnan(affine_r).any():
            right_volume_pad, right_volume = self.volumes.volume_estimate(pad_pcd_r, landmark_points_r, affine_r, interpo_depth_pcd_r)
        else:
            right_volume_pad, right_volume = 0, 0 # Landmark가 잘못 찍힌 경우 affine transform을 구하지 못하는 경우가 있어 실패

        volume = left_volume + right_volume
        volume_pad = left_volume_pad + right_volume_pad

        return volume_pad, left_volume_pad, right_volume_pad, volume, left_volume, right_volume

    def run(self, image_path):
        import os
        import glob
        import cv2

        filenames = glob.glob(image_path+'images/*') # Filename Load : List()
        valid_lst = []
        for i, filename in enumerate(filenames):
            try:
                print(i, filename)

                #### DATA LOAD ####
                bname = os.path.basename(filename) 
                image = cv2.imread(filename) # Image Load
                depth_gt_path = os.path.join(image_path, 'npy', filename.split('/')[-1].replace('img','depth').replace('png', 'npy'))
                depth_gt = np.load(depth_gt_path) # Depth(G.T.) Load
                ###################

                #### DETECTOR ####
                pred_mask, pred_info = self.detectorProcess(image, filename=bname) # Mask Image, [Pad box, keypoints]
                ###################

                #### VOLUME ####
                volume_pad, volume_pad_left, volume_pad_right, volume, left_volume, right_volume = self.volumeProcess(depth_gt, pred_mask, pred_info) # pcd
                ###################

                #### Validate Csv ####
                save_img = os.path.join('img', bname)
                save_mask = os.path.join('mask', bname.replace('img', 'mask'))
                save_point = os.path.join('point', bname.replace('img', 'point'))
                # save_stl = os.path.join('stl', bname.replace('img', 'stl').replace('png', 'stl'))
                valid_lst.append([str(i+1), save_img, save_mask,'', save_point, '', '', volume, left_volume, right_volume, '', volume_pad, volume_pad_left, volume_pad_right, ''])
                ###################

                print(f"Filename is {filename} Pad Volume: {volume_pad}, Connecte Volume {volume}, Left Volume : {left_volume} Right Volume {right_volume}")
            except:
                print('error')

        df = pd.DataFrame(valid_lst, columns=['ID', 'Image', 'Mask', 'Valid Mask', 'Landmark', 'Valid Landmark', 'Rendering', 'Volume', "Left Volume", "Right Volume", 'Valid Volume', 'pad volume', 'left pad volume', 'right pad volume', 'Valid Pad Volume'])
        df.to_csv(os.path.join(self.save_path, 'valid.csv'), index=False)


if __name__ == '__main__':
    root_path = r'/home/ljj/data/anti/validation_anti/4/'
    save_path = r'/home/ljj/data/anti/valid_4'

    app = FittingRoomAI_GT(DEBUG=False, save_path=save_path)
    app.run(image_path=root_path)