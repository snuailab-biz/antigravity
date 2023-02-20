from vtkmodules.util import numpy_support

import common
from common.common import timefn2
from detector.detector_main import AntiDetector
from estimator.estimator_main import AntiDepths
from imgproc.postprocessing import PostProcessing
from imgproc.preprocessing import PreProcessing
from reconstructor.reconstructor_main import AntiRecons


class FittingRoomAI(object):
    """
    pre-processing + detector + depth-estimator + volume-rendering + post-processing
    """

    def __init__(self, DEBUG):
        self.debug = DEBUG
        self.logger = common.get_runtime_logger()
        self.detector = AntiDetector()
        self.depths = AntiDepths()
        self.recons = AntiRecons()
        self.preproc = PreProcessing()
        self.postproc = PostProcessing()

    @timefn2
    def detectorProcess(self, images):
        '''
        pred_mask : mask부분이 색칠된 image
        pred_keypoint : Keypoints에 대해 색칠된 image
        pred_info : {'bbox': {'left': bbox, 'right': bbox}, 'landmark': {'left':Keypoints, 'right': keypoints}
        '''
        pred_mask, pred_keypoint, pred_info = self.detector.predict(images, DEBUG=self.debug)
        # pred_mask, pred_info = self.detector.predict(images, DEBUG=self.debug)

        self.logger.info("mask segmentation complete")
        return pred_mask, pred_info

    @timefn2
    def depthProcess(self, images, filename):
        pred_depth = self.depths.predict(images, filename, DEBUG=self.debug)
        self.logger.info("depth estimator complete")

        return pred_depth

    @timefn2
    def reconsProcess(self, images, pred_depths, pred_masks):
        # TODO: 포인트클라우드 정제화 기술 개발

        # mask 영역 내 pointcloud 추출
        polydatas = self.recons.masked_points_to_mesh(images, pred_depths, pred_masks, DEBUG=self.debug)
        closed_mesh = []
        # polydata -> verts, faces 추출
        for polydata in polydatas:
            verts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

            # 가상의 가슴 평면까지 ray-casting 영역 volume rendering
            closed_mesh.append(self.recons.run(verts, polydata, DEBUG=self.debug))

        self.logger.info("reconstructor complete")
        return closed_mesh

    def run(self, image_path):

        images, filename = self.preproc.image_resize_preserve_ratio(image_path=image_path)

        pred_masks, pred_infos = self.detectorProcess(images)

        pred_depths = self.depthProcess(images, filename)

        pred_mesh = self.reconsProcess(images, pred_depths, pred_masks)

        pred_voxel = self.postproc.run(pred_mesh, DEBUG=self.debug)

if __name__ == '__main__':
    image_path = r'/home/ljj/anti/fit-ai-volume-ljj/dataset/azure_kinect_pad/0203sample/images'

    app = FittingRoomAI(DEBUG=False)
    app.run(image_path=image_path)