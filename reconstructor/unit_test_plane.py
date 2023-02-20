# import vtk
# from tool.utils import *

import glob
import os

import cv2
import numpy as np
# from vtkmodules.util import numpy_support
# from imgproc import PreProcessing, PostProcessing
# import vtk
import open3d as o3d

DEBUG=True

# 데이터셋 위치
root = '/home/ljj/dataset/anti102'

def make_point_cloud_from_rgbd(color_image, depth_image, outlier=True):
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    color = o3d.geometry.Image(color_image)
    depth = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=120.0, depth_trunc=30.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole)

    if outlier:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1)

    return pcd

def getPlateProject(points):
    '''
    벡터 벡터 외적 수직 평면
    '''

    p1=points[0]
    p2=points[1]
    p3=points[2]

    r1 = p2 - p1
    r2 = p3 - p1

    Normal_Vector = np.cross(r1, r2)
    Normal_UnitVector = Normal_Vector/np.linalg.norm(Normal_Vector)

    Plate_UnitVector1 = r1/np.linalg.norm(r1);
    Plate_UnitVector2 = np.cross(Normal_UnitVector, Plate_UnitVector1)

    return Plate_UnitVector1,Plate_UnitVector2,Normal_UnitVector
    

mask_color={
        "pad_l": [33, 33, 33],
        "pad_r": [15,15,15],
        "wire_l": [11,11,11],
        "wire_r": [28,28,28],
        "landmark_t1": [26,26,26],
        "landmark_l": [29,29,29],
        "landmark_b1": [5,5,5],
        "landmark_b2": [6,6,6],
        "landmark_r": [13,13,13],
        "landmark_t2": [1,1,1],
        "landmark_c": [14,14,14]
    }

def get_point(mask, color, depth): 
    '''
    Test를 위한 G.T.로부터 뽑아낼 때 사용.
    landmark detector의 경우는 keypoints를 전달함.
    3점(기준점)이 되는 것은 color_lst를 변경하여 사용하면 됩니다.
    '''
    point_depth = np.zeros(depth.shape, dtype=np.uint8)
    
    color_lst = ['landmark_b1', 'landmark_r', 'landmark_l']
    points = []

    for c in color_lst:
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color[c])*255, dtype=np.uint8)
        gray = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 1, param2 = 1, minRadius = 0, maxRadius = 50)
        if c =='landmark_l':
            # left점이 너무 끝에 찍혀 있어 depth image상에서 가까운점이 아니라 바닥근처의 점으로 보임.
            # 이 부분은 점으로부터 pad(pad+wire) segment영역의 가장 가까운 점을 찾는 processing을 거쳐서 찾게 해야함.(현재 하드코딩 10pixel정도)
            # print(center, bottom, left)
            # find_neighbor_coord(point, segment_map)
            circles[0][0][0] +=15
        points.append((int(circles[0][0][1]), int(circles[0][0][0])))
        point = (int(circles[0][0][1]), int(circles[0][0][0]))

        # 하드코딩되어있는 부분.
        # plane 조절 값 
        gain = 5
        if c == 'landmark_l':
            point_depth[point] = depth[point] - gain
        elif c == 'landmark_r':
            point_depth[point] = depth[point] - gain
        else:
            point_depth[point] = depth[point]

    return point_depth


image_names = glob.glob(os.path.join(root,'images/*'))

for image_name in image_names:
    image_name = os.path.basename(image_name)
    image_path = os.path.join(root, 'images', image_name)
    mask_path = os.path.join(root, 'segmentations', image_name.replace('img', 'mask'))
    depth_path = os.path.join(root, 'depths', image_name.replace('img', 'depth'))
    # mask_path = '/home/ljj/dataset/anti102/segmentations/bubbledoll_80C_mask_01.png'
    # depth_path = '/home/ljj/dataset/anti102/depths/bubbledoll_80C_depth_01.png'
    '''
    다음 프로세스를 거치게 되면 resize시에 색상영역이 바뀌어 점을 찾지 못함(G.T.)
    실제 전체 프로세스 상에서는 미리 resize를 한 image에 대한 landmark를 찾아 값(landmark 좌표)으로 전달하기 때문에 상관없음.
    unit test(G.T.를 이용한)에서는 원본 크기의 이미지를 사용함.
    '''
    # preproc = PreProcessing()
    # image = preproc.image_resize_preserve_ratio(image_path=image_path)
    # mask = preproc.image_resize_preserve_ratio(image_path=mask_path)
    # depth = preproc.image_resize_preserve_ratio(image_path=depth_path)
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    depth = cv2.imread(depth_path)

    point_depth = get_point(mask, mask_color, depth)

    masked_image = image.copy()
    masked_depth = depth.copy()

    mask = mask > 0
    masked_image[mask == False] = 0
    masked_depth[masked_image[:, :, 0] == 0] = 0
    pcd = make_point_cloud_from_rgbd(image, masked_depth)

    pcd_vector = make_point_cloud_from_rgbd(image, point_depth, outlier=False)
    pcd_arr = np.asarray(pcd_vector.points)

    # 카메라 좌표계
    # plate1 = np.array([1, 0, 0])
    # plate2 = np.array([0, 1, 0])
    # norm = np.array([0, 0, 1])
    # pcd_arr[0] = np.array([0,0,0])

    # 사용자 착용기준 몸통 좌표계
    plate1, plate2, norm = getPlateProject(pcd_arr)

    # affine = np.array([[plate1[0], plate2[0], norm[0], pcd_arr[0][0]],[plate1[1],plate2[1],norm[1], pcd_arr[0][1]],[plate1[2],plate2[2],norm[2], pcd_arr[0][2]],[0,0,0, 1]])
    affine = np.array([plate1, plate2, norm, pcd_arr[0]])
    affine = np.c_[affine, np.array([0,0,0,1])].T

    # Add Plane
    plane_points = []
    scale = 0.00002
    for i in range(-60, 200):
        for j in range(-100, 100):
            point = np.array([[i*scale, j*scale, 0, 1]])
            plane_points.append(tuple(np.dot(affine, point.T)[:3].tolist()))
    # a = np.array(a)

    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(plane_points)

    p1_points = np.asarray(pcd.points)
    # p2_points = np.squeeze(np.asarray(plane_points))
    p2_points = np.asarray(pcd_plane.points)
    p3_points = np.concatenate((p1_points, p2_points), axis=0)

    p1_color = np.asarray(pcd.colors)
    p2_color = np.ones(p2_points.shape) * [251, 206, 177]/255
    p3_color = np.concatenate((p1_color, p2_color), axis=0)
    pcd_plane.colors = o3d.utility.Vector3dVector(p2_color)

    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(p3_points)
    pcd_final.colors = o3d.utility.Vector3dVector(p3_color)

    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd_plane])
    o3d.visualization.draw_geometries([pcd_final])

