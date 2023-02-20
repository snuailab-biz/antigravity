import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pymeshlab
from pyntcloud import PyntCloud

from common import timefn2
from tool import vtk_utils


@timefn2
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


@timefn2
def rgbd_to_volume(rgb, depth):

    depth[rgb[:, :, 0] == 0] = 0

    color = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=10.0, depth_trunc=1000.0 )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
    volume = get_volume_from_approximation(pcd)

    return volume

@timefn2
def pcd_to_voxel(pcd:o3d.geometry.PointCloud, voxel_size=0.01, DEBUG=False):
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    if DEBUG:
        o3d.visualization.draw_geometries([voxel_grid])
    return voxel_grid

@timefn2
def poisson_reconstruction(point_cloud: o3d.geometry.PointCloud, **kwargs):
    point_cloud.compute_convex_hull()
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(20)

    bbox = point_cloud.get_axis_aligned_bounding_box()

    config = kwargs.get("reconstruction", dict())
    depth = config.get("possion_depth", 10)
    scale = config.get("poisson_scale", 1.1)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth, width=0, scale=scale, linear_fit=False)[0]
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()
    refined_mesh = mesh.crop(bbox)

    return refined_mesh


@timefn2
def save_refined_mesh(color_image: np.ndarray, depth_image: np.ndarray, nb_neighbors: int = 10, std_ratio: float = 0.5,
                      scale_factor=1e4, camera_params: dict = None):
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    if camera_params:
        pinhole.set_intrinsics(
            color_image.shape[0],
            color_image.shape[1],
            camera_params["fx"],
            camera_params["fy"],
            camera_params["cx"],
            camera_params["cy"],
        )

    depth_image = cv2.medianBlur(depth_image, 3)

    color = o3d.geometry.Image(color_image)
    depth = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=100.0, depth_trunc=30.0)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    downpcd = pcd.uniform_down_sample(10)
    cl, _ = downpcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    pcd_points = np.asanyarray(cl.points) * scale_factor

    save_point_to_mesh(pcd_points)

    return pcd_points


@timefn2
def save_point_to_mesh(points, filename='temp'):
    ms = pymeshlab.MeshSet()
    pcd_mesh = pymeshlab.Mesh(points)
    ms.add_mesh(pcd_mesh)
    ms.generate_simplified_point_cloud()
    ms.generate_surface_reconstruction_ball_pivoting(clustering=20, creasethr=60)
    ms.meshing_close_holes(maxholesize=15, newfaceselected=False)
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)
    ms.save_current_mesh("{}.stl".format(filename))


@timefn2
def ols_squre(x: np.ndarray, y: np.ndarray):
    bias = np.ones_like(x)
    x = np.vstack((bias, x, x ** 2))
    inv = x @ x.T
    inv = np.linalg.pinv(inv)
    beta_hat = inv @ x
    beta_hat = beta_hat @ y

    return beta_hat

@timefn2
def get_volume_from_approximation(pcd: o3d.geometry.PointCloud) -> float:
    """
    Get volume from point cloud with approximation.
    1. Get approximated squared equation from points via least square methods. (in x, z axis)
    2. Get approximated circle from points via Min Enclosing Circle methods. (in x, y axis)
    """

    def func(x, beta):
        return beta[0] + beta[1] * x + beta[2] * x ** 2

    def get_length_and_tip(c, b, a):
        tip = func(-b / (2 * a), beta)
        length = np.sqrt((b / a) ** 2 - 4 * (c / a))
        return length, tip

    points = np.asarray(pcd.points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ys = np.array([y.min(), y.max(), y[np.argmax(z)]])
    zs = np.array([z[np.argmin(y)], z[np.argmax(y)], z.max()])
    beta = ols_squre(ys, zs)

    length, tip = get_length_and_tip(*beta)

    L = list(zip(x, y))
    ctr = np.array(L).reshape((-1, 1, 2)).astype(np.int32)
    r = cv2.minEnclosingCircle(ctr)[-1]

    return tip

@timefn2
def show_binvox(binvox):
    """
        with open(file_path, 'rb') as f:
        m1 = binvox_rw.read_as_3d_array(f)
    """

    show = vtk_utils.show_actors
    p2a = vtk_utils.polydata2actor

    vox = binvox.data
    vtk_image = vtk_utils.convert_numpy_vtkimag(vox)
    restored_polydata = vtk_utils.convert_voxel_to_polydata(vtk_image, True)

    show([ p2a(restored_polydata)])

@timefn2
def pts_to_voxel(points:np.asarray, ndim=128, translate=[0, 0, 0], scale=1.0, axis_order='xyz', DEBUG=False):

    points = points-points.min(axis=0)
    points = (points/points.max())*float(ndim-1)

    noisy_points = np.where(points[:, 2] < np.median(points[:, 2]))
    pts = np.delete(points, noisy_points, axis=0)

    new_pts = {'x': pts[:, 0], 'y': pts[:, 1], 'z': pts[:, 2]}
    cloud = PyntCloud(pd.DataFrame(new_pts))

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=ndim, n_y=ndim, n_z=ndim)
    voxelgrid = cloud.structures[voxelgrid_id]

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((ndim, ndim, ndim)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True

    return voxel

