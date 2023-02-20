import glob
from tool.utils import *


if __name__ == '__main__':

    DEBUG = True

    data_path = './dataset/dataset_test_iphone'
    result_model_path = 'anti_densenet161_pad_nyu_256'

    img_path = '{}/images'.format(data_path)
    seg_path = '{}/segmentaions'.format(data_path)
    pred_depth_dir = glob.glob('{}/result_{}/*.png'.format(data_path, result_model_path)) or glob.glob('{}/result_{}/*.jpg'.format(data_path, result_model_path))

    result_root_dir = './result'
    pred_mesh_dir = os.path.join(result_root_dir, result_model_path)

    if not os.path.isdir(result_root_dir):
        os.mkdir(result_root_dir)

    if not os.path.isdir(pred_mesh_dir):
        os.mkdir(pred_mesh_dir)

    for filepath in pred_depth_dir:

        filename = filepath.split('\\')[-1]
        img = cv2.imread('{}/{}'.format(img_path, filename))
        pred_depth = cv2.imread(filepath)

        pcd = make_point_cloud_from_rgbd(img, pred_depth)
        if DEBUG:
            cv2.namedWindow(filename)  # create a named window
            cv2.moveWindow(filename, 150, 150)
            cv2.imshow(filename, img)
            o3d.visualization.draw_geometries([pcd], width=1280, height=720, left=700, top=200)
            cv2.destroyAllWindows()

        # voxel
        voxel_grid = pcd_to_voxel(pcd, DEBUG=True)


