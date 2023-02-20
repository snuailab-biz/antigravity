import vtk

# from tool.utils import *
import open3d as o3d
import numpy as np
from functools import reduce
from scipy.spatial import Delaunay
import scipy

DEBUG = True 

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

class VolumeEstimateGT(object):
    def __init__(self, DEBUG=False):
        # TODO: landmark info -> plane 가상평면 생성
        self.DEBUG=DEBUG
    
    def volume_preprocessing(self, depth, mask, keypoint, locate='left'):
        landmark = keypoint[locate][1:] # left : t, l, b, r
        label_color = mask_color['pad_l'] if locate=='left' else mask_color['pad_r']

        # ===================== Get Ground Truth ===================== #
        pad_depth = self.get_pad_reduce(depth, mask.copy(), label_color[0], reduce_value=5) # reduce_value만큼 축소된 마스크 영역에 대해 depth값 추출/
        pad_pcd = self.improve_Depth2PointCloud(pad_depth, 0.001) # masked depth 부분을 point cloud로 변환, 0.001은 이후 계산할 경우 값이 너무 커지기 때문에 축소했을 뿐.

        plane_point, points = self.get_point(pad_depth, landmark) # points : [l, b, r]
        landmark_points = [pad_pcd[p] for p in points] # 각 포인트를 point cloud로 변환 (affine transform을 계산하기 위해 사용)
        affine = self.get_AFM(landmark_points)

        # ===================== interpolation ===================== #
        center_point = self.get_center_point(mask, label_color) # mask영역의 중심점 추출
        points.append(center_point) # points : [l, b, r], center 추가

        interpo_depth, MEA, status = self.improve_interpolate(pad_depth, points) # MEA : Error율, status는 연장이 되었는지 안되었는지 판단.
        interpo_depth = self.except_masked_depth(interpo_depth, pad_depth) # 연장된 영역이 mask영역에 포함되어 있으므로 제거
        interpo_depth_pcd = self.improve_Depth2PointCloud(interpo_depth, 0.001) # point cloud로 변환하는 과정으로 연장곡선에 대해서도 진행

        return pad_pcd, landmark_points, affine, interpo_depth_pcd

    def except_masked_depth(self, depth,mask):
        temp_depth_img = np.array(depth.copy(), dtype=np.uint16)
        mask_inex = np.where(mask>0);
        for (y,x) in zip(mask_inex[0],mask_inex[1]):
            temp_depth_img[y,x] = 0;
        return temp_depth_img;

    def volume_estimate(self, pcd_arr, landmark_points, affine, interpo_arr=None, DEBUG=False):
        """
        Volume Estimate Algorithm
        """
        ########## 좌표계
        plate_TM = o3d.geometry.TriangleMesh.create_coordinate_frame()
        plate_TM.scale(0.1, center=(0, 0, 0))
        plate_TM.rotate(affine[:3,:3], center=(0, 0, 0))
        plate_TM.translate(landmark_points[0])
        ########## 좌표계

        ########## 가상평면보다 밑으로 있는 값들 제거
        masked_depth_pcd = pcd_arr[np.where(np.dot(pcd_arr-landmark_points[0], affine[:3, 2].T) > 0.0)]
        interpo_depth_pcd = interpo_arr[np.where(np.dot(interpo_arr-landmark_points[0], affine[:3, 2].T) > 0.0)]
        ########## 가상평면보다 밑으로 있는 값들 제거

        ########## pcd array에 대해 open3d point cloud 생성
        depth_img_stream=masked_depth_pcd.reshape(-1,3)
        depth_img_stream=depth_img_stream[np.where(depth_img_stream[:,2] > 0.1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_img_stream)

        # Pad 영역 point cloud
        inlier_rad_pcd = o3d.geometry.PointCloud()
        inlier_rad_pcd.points = o3d.utility.Vector3dVector(depth_img_stream)

        # 연장 곡선 영역 point cloud
        interpo_pcd = o3d.geometry.PointCloud()
        interpo_depth_img_stream=interpo_depth_pcd.reshape(-1,3);
        interpo_depth_img_stream=interpo_depth_img_stream[np.where(interpo_depth_img_stream[:,2]>0.1)];
        interpo_pcd.points = o3d.utility.Vector3dVector(interpo_depth_img_stream)

        # pad 영역 point cloud를 보기 위한 색상 정보 전달
        color_temp = np.zeros((interpo_depth_img_stream.shape))+[0,255,0]
        color_temp/=255.0;
        interpo_pcd.colors = o3d.utility.Vector3dVector(color_temp)
        ########## pcd array에 대해 open3d point cloud 생성
        
        # outlier 제거, depth gt가 정확하다면 제거하지 않아도 됨. noise가 있기 때문에 제거하는 것.
        for i in range(1):
            cl, ind = inlier_rad_pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=1.0);
            inlier_rad_pcd = inlier_rad_pcd.select_by_index(ind);


        inlier_rad_pcd = inlier_rad_pcd.voxel_down_sample(voxel_size=0.001);
        interpo_pcd = interpo_pcd.voxel_down_sample(voxel_size=0.001);

        ########## 카메라 방향에서의 point cloud를 affine transform과 기준이 되는 점을 통해 rotate, translate
        inlier_rad_pcd.translate(-landmark_points[0])
        inlier_rad_pcd.rotate(affine[:3,:3].T, center=(0, 0, 0))
        inlier_rad_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        interpo_pcd.translate(-landmark_points[0])
        interpo_pcd.rotate(affine[:3,:3].T, center=(0, 0, 0))
        interpo_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Depth Scale에 따라 변경되는 부분이며, 현재 camera parameter에 대해서는 
        alpha = 0.018

        ############################## Pad Volume Estimate #################################
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(inlier_rad_pcd)
        mesh_inlier = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(inlier_rad_pcd, alpha, tetra_mesh, pt_map)
        mesh_inlier.compute_vertex_normals()
        # downpdc_inlier = mesh_inlier.sample_points_uniformly(number_of_points=5000) # mesh에서 5000개의 포인트만 추출. (더 많이 할 경우 하드웨어가 받쳐주질 않아서 5000으로 설정함.)
        # volume_inlier = self.get_volume(downpdc_inlier.points)
        # print(f"The volume of the pad is : {round(volume_inlier, 9)*1000000}cc")

        # 평면으로부터 mesh상 가장 가까운 min point에 대해서만 추출
        inlier_raycast_points = self.PlatRaycast_For_BraMesh(  mesh = mesh_inlier,
                                                center = [0,0,0.1], 
                                                direct = [0,0,-1],
                                                resolution = 0.001,
                                                range_x=[-0.5,0.5],
                                                range_y=[-0.5,0.5]);
        inlier_raycast_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_raycast_points))

        cl, ind_inlier = inlier_raycast_pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0);
        inlier_raycast_pcd = inlier_raycast_pcd.select_by_index(ind_inlier);

        volume_inlier_ray = self.get_volume(inlier_raycast_pcd.points)
        print(f"The volume of the pad is : {round(volume_inlier_ray, 9)*1000000}cc")
        volume_pad = round(volume_inlier_ray, 9)*1000000
        
        ############################## Connected Volume Estimate #################################
        connect_pcd = inlier_rad_pcd+interpo_pcd

        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(connect_pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(connect_pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        # downpdc = mesh.sample_points_uniformly(number_of_points=5000) # mesh에서 5000개의 포인트만 추출. (더 많이 할 경우 하드웨어가 받쳐주질 않아서 5000으로 설정함.)
        # volume1 = self.get_volume(downpdc.points)
        # print(f"The volume of the pad is : {round(volume1, 9)*1000000}cc")

        # 평면으로부터 mesh상 가장 가까운 min point에 대해서만 추출
        raycast_points = self.PlatRaycast_For_BraMesh(  mesh = mesh,
                                                center = [0,0,0.1], 
                                                direct = [0,0,-1],
                                                # center = [0,0,-0.1], # 
                                                # direct = [0,0,1],
                                                resolution = 0.001,
                                                range_x=[-0.5,0.5],
                                                range_y=[-0.5,0.5]);

        raycast_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raycast_points))

        cl, ind = raycast_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0);
        raycast_pcd = raycast_pcd.select_by_index(ind);

        volume2 = self.get_volume(raycast_pcd.points)
        volume_connect = round(volume2, 9)*1000000


        print(f"The volume of the pad is : {volume_connect}cc, {volume_pad}")

        if self.DEBUG:
            o3d.visualization.draw_geometries([mesh_inlier, inlier_raycast_pcd], mesh_show_back_face=True)
            o3d.visualization.draw_geometries([inlier_raycast_pcd, inlier_rad_pcd])

            o3d.visualization.draw_geometries([mesh, raycast_pcd], mesh_show_back_face=True)
            o3d.visualization.draw_geometries([raycast_pcd, inlier_rad_pcd, interpo_pcd])
    
        return volume_pad, volume_connect
    
    def get_volume(self, points):
        xyz = points
        xy_catalog = []
        for point in xyz:
            xy_catalog.append([point[0], point[1]])

        xy_catalog = np.array(xy_catalog);
        tri = scipy.spatial.Delaunay(xy_catalog)

        surface = o3d.geometry.TriangleMesh()
        surface.vertices = o3d.utility.Vector3dVector(xyz)
        surface.triangles = o3d.utility.Vector3iVector(tri.simplices) 
        
        def volume_under_triangle(triangle):
            p1, p2, p3 = triangle
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            x3, y3, z3 = p3
            return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)

        def get_triangles_vertices(triangles, vertices):
            triangles_vertices = []
            for triangle in triangles:
                new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
                triangles_vertices.append(new_triangles_vertices)
            return np.array(triangles_vertices)
        volume = reduce(lambda a, b:  a + volume_under_triangle(b), get_triangles_vertices(surface.triangles, surface.vertices), 0)

        return volume

    def bra_fitting_func(self,fit_data_xy,limit_x=(-4,4),limit_y=(0,4),resolution = 1.0,func_name = "cubic", avg_Parameters = [], avg_weight = 0.0):
        # writer : dkko
        # date   : 20230119
        # e-mail : dkko@snuailab.ai

        # limit_x    = (-maximum_data_x, last_data_x )
        # limit_y    = (plate_offset ,bra_max_height)
        # resolution = depth image pixel size(defualt = 1)
        # func_name  = fitting function name

        x=fit_data_xy[0];
        y=fit_data_xy[1];

        # 함수
        if(func_name=="linear"):
            def func(x, a, b): 
                return a*(x) + b
            initialParameters = np.array([1.0, 1.0 ]);

        if(func_name=="bra_quadratic"):
            def func(x, a, b, c ): 
                return -1.0*np.abs(a)*(x*x) + b*(x) + c
            initialParameters = np.array([1.0, 1.0 ,1.0]);
        if(func_name=="quadratic"):
            def func(x, a, b, c ): 
                return a*(x*x) + b*(x) + c
            initialParameters = np.array([1.0, 1.0 ,1.0]);

        if(func_name=="sin"):
            def func(x, a, b ): 
                return np.sin(a*x) + b;
            initialParameters = np.array([1.0, 1.0]);

        if(func_name=="bra"):
            def func(x, a, b, c ,d ): 
                return -1.0*np.abs(a)*(x*x*x) + b*(x*x) + c*(x) + d;
            initialParameters = np.array([1.0, 1.0 ,1.0, 1.0]);

        if(func_name=="cubic"):
            def func(x, a, b, c ,d ): 
                return a*(x*x*x) + b*(x*x) + c*(x) + d
            initialParameters = np.array([1.0, 1.0 ,1.0, 1.0]);
        if(func_name=="log"):
            def func(x, a, b, c): # x-shifted log
                return a*np.log(x + b) + c
            initialParameters = np.array([1.0, 1.0 ,1.0]);
        if(func_name=="gaussian"):
            def func(x, amp, cen, wid):
                return amp * np.exp(-(x-cen)**2 / wid)
            initialParameters = np.array([1.0, 1.0 ,1.0]);

        # 피팅
        fittedParameters, pcov = scipy.optimize.curve_fit(func, x, y, initialParameters,check_finite = True, bounds=(0.0,1.0));
        # modelPredictions = func(x, *fittedParameters) 
        # absError = modelPredictions - y;
        
        # x 데이터 생성
        x=np.arange(limit_x[0], limit_x[1], resolution);


        # 현재 데이터 모델 + 평균 모델 가중
        if (len(avg_Parameters)!=0):
            y_hat = func(x, *((1.0-avg_weight)*fittedParameters) + (avg_weight*avg_Parameters));
        # 현재 데이터 모델
        else:
            y_hat = func(x, *fittedParameters);


        return x[np.argwhere((y_hat<limit_y[1]) & (y_hat>limit_y[0]) )], y_hat[np.argwhere((y_hat<limit_y[1]) & (y_hat>limit_y[0]))],fittedParameters;


    def improve_interpolate(self, depth, points, r_resulotion = 1.0,theta_resulotion = 1.0,depth_type = 'uint16', all = False):
        # r_resulotion demension is pixel
        # theta_resulotion demension is degree
        # p1,p2,p3,cp are center-point 
        if(depth_type == 'uint16'):
            depth_resolution = 2**16;
        if(depth_type == 'uint32'):
            depth_resolution = 2**32;

        p1 = (points[0][1], points[0][0])
        cp = (points[3][1], points[3][0])
        p1y,p1x = points[0]
        p2y,p2x = points[1]
        p3y,p3x = points[2]
        cpy,cpx = points[3]

        # 속옷 중심점과 각 랜드마크의 각도(rad) 계산
        theta1= np.arctan2(( p1y - cpy) , (p1x - cpx));
        theta2= np.arctan2(( p2y - cpy) , (p2x - cpx));
        theta3= np.arctan2(( p3y - cpy) , (p3x - cpx));

        # 피팅할 원의 반지름 거리 계산
        radius=np.linalg.norm(np.array(cp)-np.array(p1));
        # print("radius:",radius)
        # print(p1x , cpx);


        # pixel 별 반지름 데이터 생성
        r_range = np.linspace(0.0,radius,num=int(radius/r_resulotion));

        # 랜드마크에 따라 피팅 위치 계산
        theta_range = [];

        # theta1 += 2.0*np.pi;  
        # theta2 += 2.0*np.pi;
        # theta3 += 2.0*np.pi;

        # 피팅 위치 계산
        d1 = np.abs(theta3-theta2);
        d2 =  np.abs(theta1-theta2);
        if(d1 > np.pi):
            d1 = 2* np.pi - d1;
        if(d2 > np.pi):
            d2 = 2* np.pi - d2;

        except_range = d1+d2;
        move_angle_sol = np.pi*2 - except_range;

        if(np.sin(theta2)>0.0):
            theta_range = list(np.arange(theta1,theta1+move_angle_sol, (np.pi/180.0)*theta_resulotion));
        else:
            theta_range = list(np.arange(theta3,theta3+move_angle_sol, (np.pi/180.0)*theta_resulotion));

        if(all):
            theta_range = list(np.arange(0 ,np.pi*2, (np.pi/180.0)*10.0));
        
        # print("theta_range",theta_range, )
        # print("theta1 : ",np.rad2deg(theta1));
        # print("theta2 : ",np.rad2deg(theta2));
        # print("theta3 : ",np.rad2deg(theta3));    
        
            
        # 연장 부분의 평균 모델의 형상 변수
        avg_parm = [];

        # 연장한 부분을 저장할 depth 이미지 변수
        interpolation_depth = np.zeros((depth.shape));
        
        success_fit = True;
        
        h_pad = 5;

        
        # cp위치로 부터 반지름 크기와 방향으로 x,y 좌표 계산하여 피팅함수 계산
        for theta_ in theta_range:
            temp_list_y = [];
            temp_list_x = [];
            for r_ in r_range:
                temp_x = cpx + int(r_ *np.cos(theta_)) ;
                temp_y = cpy + int(r_ *np.sin(theta_)) ;
                depth_patch = depth[temp_y-h_pad:temp_y+h_pad,temp_x-h_pad:temp_x+h_pad];
                b_idx=np.where(depth_patch > 0.0)
                if(b_idx[0].__len__()>0):
                    depth_patch_avg=depth_patch[b_idx].min();
                    depth_value=depth_patch_avg/depth_resolution;
                    temp_list_y.append(depth_value);
                    temp_list_x.append(r_);
            
            # plt.plot(temp_list_x,temp_list_y);
            # plt.show()

            temp_list_x=np.array(temp_list_x)
            temp_list_y=np.array(temp_list_y)
            mask_background=np.where(temp_list_y>0.0)

            temp_list_x = temp_list_x[mask_background]
            temp_list_y = temp_list_y[mask_background]

            
            # 불연속면에 대한 신호 처리
            temp_list_x_check=[];
            temp_list_y_check=[];
            pre_diff_y = temp_list_y[1]-temp_list_y[0];
            pre_y = temp_list_y[0];
            for y_idx in range(len(temp_list_y)-1):
                diff_y=temp_list_y[y_idx+1]-temp_list_y[y_idx]
                diff_x=temp_list_x[y_idx+1]-temp_list_x[y_idx]
                
                alpha = 0.3;
                # LPF(Low-Pass-Filter)
                pre_diff_y = ((1.0-alpha)*diff_y) + (alpha * pre_diff_y);
                pre_y = ((1.0-alpha)*temp_list_y[y_idx]) + (alpha * pre_y);
                
                dy_dx=pre_diff_y/diff_x;
                if(pre_diff_y<=0.0):
                    temp_list_x_check.append(temp_list_x[y_idx]);
                    temp_list_y_check.append(pre_y);
                
            #plt.plot(temp_list_x_check,temp_list_y_check);
            # 연장 피팅 함수 계산
            if ((len(temp_list_x) < 6) or (len(temp_list_y_check)<6)):
                continue
            x_hat,y_hat,parm=self.bra_fitting_func(  np.array([temp_list_x_check,temp_list_y_check]),
                                                limit_x=(0,int(radius+50)), 
                                                limit_y=(0,1),
                                                func_name="bra",
                                                avg_Parameters=[],
                                                avg_weight=0.0);    
            avg_parm.append(parm);
            
            try:
                # Depth 값을 BGR로 Encoding
                for idx_,r_ in enumerate(x_hat):
                    temp_x = cpx + int(r_ *np.cos(theta_)) ;
                    temp_y = cpy + int(r_ *np.sin(theta_)) ;    
                    interpolation_depth[temp_y,temp_x] = y_hat[idx_]*depth_resolution;
            except:
                success_fit = False
                pass;
            

        # 피팅 결과의 평균값을 계산한다. 
        avg_parm = np.array(avg_parm).mean(axis=0);
        return interpolation_depth, avg_parm, success_fit


    
        
    def PlatRaycast_For_BraMesh(self, mesh,center=[0,0,0],direct=[0,0,1], resolution = 0.1,range_x = [-1.0,1.0],range_y = [-1.0,1.0]):
        min_x = range_x[0];
        max_x = range_x[1];
        num_x = int((max_x-min_x)/resolution);
        min_y = range_y[0];
        max_y = range_y[1];
        num_y = int((max_y-min_y)/resolution);
        x_data = np.linspace(min_x,max_x,num_x);
        y_data = np.linspace(min_y,max_y,num_y);

        ray_list = [];

        # ray들의 중심
        cp_x = center[0];
        cp_y = center[1];
        cp_z = center[2];

        # ray들의 방향
        dir_x = direct[0];
        dir_y = direct[1];
        dir_z = direct[2];

        # ray vector들 계산
        for x_ in x_data:
            for y_ in y_data:

                ref_x = cp_x + x_;
                ref_y = cp_y + y_;
                ref_z = cp_z;
            
                ray_list.append([ref_x,ref_y,ref_z,dir_x,dir_y,dir_z]);
        
        # ray 생성
        rays = o3d.core.Tensor(ray_list,dtype=o3d.core.Dtype.Float32);

        # Raycasting용 Scene 생성
        scene = o3d.t.geometry.RaycastingScene()

        # Scene에 mesh 적용
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        # mesh에 Ray 조사
        ans = scene.cast_rays(rays);
        
        # mesh와 충돌한 Ray 검출
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))

        # 충돌된 ray 위치 리턴
        return points.numpy();


    def get_center_point(self, mask, label):
        mask_index= np.where(mask==label)

        # 가로,세로 인덱스의 중심위치 계산함.
        mean_y_idx=int(mask_index[0].mean())
        mean_x_idx=int(mask_index[1].mean())
        
        # mask 추출한 위치에 이미지화
        # mask[mean_y_idx,mean_x_idx] = 255;
        # cv2.imshow("mask_img",mask);
        # cv2.waitKey(0);
        return (mean_y_idx,mean_x_idx)


    def get_pad_reduce(self, depth, mask, color, reduce_value=12):

        temp_mask = mask
        background_mask = np.zeros(depth.shape, dtype=np.uint16)

        mask_index = np.where(temp_mask==color);
        # if(len(mask_index[0]) <= 0):
            # continue
        

        # 가로,세로 라벨의 중심위치 계산함.
        center_y_idx=int(mask_index[0].mean());
        center_x_idx=int(mask_index[1].mean());
        
        #라벨의 최외각 위치 계산
        min_y_idx = mask_index[0].min();
        max_y_idx = mask_index[0].max();
        min_x_idx = mask_index[1].min();
        max_x_idx = mask_index[1].max();

        #라벨 축소 작업
        range_y_idx = max_y_idx-min_y_idx;
        range_x_idx = max_x_idx-min_x_idx;

        reduce_range_y_idx = range_y_idx - int(reduce_value*2);
        reduce_range_x_idx = range_x_idx - int(reduce_value*2);

        #선형 축소 방정식 계산
        reduce_mask_index_y = np.int64((mask_index[0]-center_y_idx)*(reduce_range_y_idx/range_y_idx)) + center_y_idx;
        reduce_mask_index_x = np.int64((mask_index[1]-center_x_idx)*(reduce_range_x_idx/range_x_idx)) + center_x_idx;

        #축소된 라벨 붙여 넣기 
        # background_mask[(reduce_mask_index_y,reduce_mask_index_x,mask_index[2])] = depth[mask_index[:2]];
        background_mask[reduce_mask_index_y, reduce_mask_index_x] = depth[reduce_mask_index_y, reduce_mask_index_x];


        return background_mask

    # def get_pad(self, depth, mask, color):
    #     masked_depth=np.zeros(depth.shape);
    #     mask_inex = np.where(mask==color);
    #     masked_depth[mask_inex[0], mask_inex[1]] = depth[mask_inex[0], mask_inex[1]]

    #     return masked_depth

    def get_point(self, mask, keypoint):
        point_depth = np.zeros(mask.shape, dtype=np.uint8)
        points = [] 
        for p in keypoint:
            point=[int(p[1]), int(p[0])]
            point = self.get_near_point(mask, point)
            point_depth[point] = mask[point]
            points.append(point)
        return point_depth, points

    def get_near_point(self, depth, pos=[0,0]):
        depth_arg = np.argwhere(depth>0)
        distance = np.linalg.norm(depth_arg - pos,axis=1)
        min_arg = np.argmin(distance)
        near_point = tuple(depth_arg[min_arg].tolist())
        return near_point

    def improve_Depth2PointCloud(self, depth_uint16, scale=0.001):
        # realsense 카메라(안티그래비티) Depth 이미지를 PointCloud로 변환하는 함수
        
        fx = 607.62634277;
        fy = 607.40606689;
        cx = 641.01501465;
        cy = 366.62744141;
        
        height,width = depth_uint16.shape[:2];
        CameraIntrinsic=o3d.camera.PinholeCameraIntrinsic();
        CameraIntrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        o3d_depth_uint16=o3d.geometry.Image(depth_uint16);
        pcd=o3d.geometry.PointCloud.create_from_depth_image(    o3d_depth_uint16,
                                                                CameraIntrinsic,
                                                                depth_scale=1.0/scale,
                                                                depth_trunc=1.0/scale,
                                                                project_valid_depth_only=False);
        return np.asarray(pcd.points).reshape(height,width,-1);

    def get_AFM(self, points):
        p1=points[0]
        p2=points[1]
        p3=points[2]
        r1 = p2 - p1
        r2 = p3 - p1
        Normal_Vector = np.cross(r1, r2)

        normal_unit = Normal_Vector/np.linalg.norm(Normal_Vector)

        plate1 = r1/np.linalg.norm(r1)
        plate2 = np.cross(normal_unit, plate1)

        affine = np.array([plate1, plate2, normal_unit*-1, p1])
        affine = np.c_[affine, np.array([0,0,0,1])].T

        return affine