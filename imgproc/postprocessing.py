import numpy as np
from vtkmodules.util import numpy_support

from common.common import timefn2
from tool import vtk_utils

show = vtk_utils.show_actors
p2a = vtk_utils.polydata2actor


class PostProcessing(object):
    def __init__(self):
        self.postporc = 'postprocessing'
        self.voxel_spacing = np.array([0.2, 0.2, 0.2], dtype=np.float32)

    @timefn2
    def run(self, polydatas, DEBUG=False):

        for polydata in polydatas:

            max_padding = np.max(self.voxel_spacing) * 1.5

            axes = vtk_utils.get_axes(10)
            vox, ctr, vox_origin = vtk_utils.polydata2voxelization_withpad(polydata, self.voxel_spacing,
                                                                           return_origin=True, padding=max_padding)

            if DEBUG:
                threshold = 123
                show([vtk_utils.numpyvolume2vtkvolume(vox, threshold), axes])

                # 복셀 데이터를 나중에 복구하기 위한 4x4 정보
                t_vox_to_src = vtk_utils.myTransform()
                t_vox_to_src.Scale(self.voxel_spacing)
                t_vox_to_src.Translate(-vox_origin)

                vox2src = t_vox_to_src.convert_np_mat()
                world_origin = vtk_utils.apply_trasnform_np(np.zeros([1, 3]), vox2src)[0]

                # voxel 데이터를 polydata로 복구. numpy 배열 ---> vtk 배열 ---> vtk배열을 vtk 알고리즘(마칭큐브) 사용해서 복구(threhold-123)
                vtk_image = vtk_utils.convert_numpy_vtkimag(vox)
                restored_polydata = vtk_utils.convert_voxel_to_polydata(vtk_image, threshold)

                show([p2a(restored_polydata), p2a(polydata)])

                # 복구한 polydata, 다시 원좌표계로 변환
                print(restored_polydata.GetNumberOfPoints(), polydata.GetNumberOfPoints())
                restored_polydata_in_src = vtk_utils.apply_transform_polydata(restored_polydata, t_vox_to_src)
                show([p2a(restored_polydata_in_src), p2a(polydata)])

                # point 클라우드 확인
                src_pts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
                t_src_2_vox = t_vox_to_src.GetInverse().convert_np_mat()
                vox_pts = vtk_utils.apply_trasnform_np(src_pts, t_src_2_vox)

                # TODO : create_points_actor error 확인 필요
                # show([ vtk_utils.numpyvolume2vtkvolume(vox, threshold), vtk_utils.create_points_actor(vox_pts)])

            return vox