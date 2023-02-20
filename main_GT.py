import common
from common.common import timefn2
from volume_estimate.volume_estimate_mainGT import VolumeEstimateGT
from detector.detector_main import AntiDetector

import numpy as np
import open3d as o3d

import pandas as pd
import csv


class FittingRoomAI_GT(object):

    def __init__(self, DEBUG):
        self.logger = common.get_runtime_logger()
        self.detector = AntiDetector(DEBUG)
        self.volumes = VolumeEstimateGT(DEBUG)

    @timefn2
    def detectorProcess(self, image):
        '''
        pred_mask : mask부분이 색칠된 image
        pred_keypoint : Keypoints에 대해 색칠된 image
        pred_info : {'bbox': {'left': bbox, 'right': bbox}, 'landmark': {'left':Keypoints, 'right': keypoints}
        '''
        pred_mask, pred_keypoint, pred_info = self.detector.predict(image)

        self.logger.info("mask segmentation complete")
        return pred_mask, pred_info


    @timefn2
    def volumeProcess(self, pred_depth, pred_mask, pred_info):
        # keypoints (Left, Bottom, Right)를 이용하여 afm 정의
        # run : output volume
        pad_pcd_l, landmark_points_l, affine_l, interpo_depth_pcd_l = self.volumes.volume_preprocessing(pred_depth, pred_mask, pred_info['landmark'], locate='left')
        pad_pcd_r, landmark_points_r, affine_r, interpo_depth_pcd_r = self.volumes.volume_preprocessing(pred_depth, pred_mask, pred_info['landmark'], locate='right')
        if not np.isnan(affine_l).any():
            left_volume, left_mesh = self.volumes.volume_estimate(pad_pcd_l, landmark_points_l, affine_l, interpo_depth_pcd_l)
        else:
            left_volume, left_mesh = 0, None

        if not np.isnan(affine_r).any():
            right_volume, right_mesh = self.volumes.volume_estimate(pad_pcd_r, landmark_points_r, affine_r, interpo_depth_pcd_r)
        else:
            right_volume, right_mesh = 0, None

        # mesh = left_mesh + right_mesh # Left Mesh + Right Mesh
        # o3d.io.write_triangle_mesh('mesh.stl', mesh) # Mesh save
        volume = left_volume + right_volume

        return volume, left_volume, right_volume

    def run(self, image_path):
        import os
        import glob
        import cv2
        filenames = glob.glob(image_path+'/*') # Filename Load : List()
        valid_lst = []
        for i, filename in enumerate(filenames):
            print(i)
            print(filename)
            image = cv2.imread(filename) # Image Load
            depth_gt_path = os.path.join(image_path.replace('images', 'npy'), filename.split('/')[-1].replace('img','depth').replace('png', 'npy'))
            depth_gt = np.load(depth_gt_path) # Depth(G.T.) Load

            pred_mask, pred_info = self.detectorProcess(image) # Mask Image, [Pad box, keypoints]
            volume, left_volume, right_volume = self.volumeProcess(depth_gt, pred_mask, pred_info) # pcd

            print(f"Filename is {filename} Pad Volume: {volume}, Left Volume : {left_volume} Right Volume {right_volume}")


if __name__ == '__main__':
    image_path = r'/home/ljj/data/anti/test/images'

    app = FittingRoomAI_GT(DEBUG=True)
    app.run(image_path=image_path)