import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from open3d import geometry
import scipy
from numpy import matrix
from numpy.linalg import solve
from functools import reduce


# 파라메터 [1]
#more_shape = -1000.0* (1e-10); 작은 가슴
#more_shape = 200.0* (1e-10) ;# 큰가슴

# 파라메터 [2]
more_shape = 1000.0* (1e-10) ;# 큰가슴



# def func(x,  cx, cy, r):
#         return np.sqrt((r**2)+((x-cx)**2)) + cy;
    
# def circfit(xs, ys):
#     a = matrix([[x,y,1.] for x,y in zip(xs, ys)])
#     b = matrix([[-(x*x + y*y)] for x,y in zip(xs, ys)])
#     res = solve(a,b)
#     xc = -0.5 * res.item(0)
#     yc = -0.5 * res.item(1)
#     r = (xc*xc + yc*yc - res.item(2))**0.5

#     return xc,yc,r

# circfit([0,1,2],[2,1,2])
# exit()

def get_surface(points):
    xyz = points
    xy_catalog = []
    for point in xyz:
        xy_catalog.append([point[0], point[1]])

    xy_catalog = np.array(xy_catalog);
    tri = scipy.spatial.Delaunay(xy_catalog)

    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(xyz)
    surface.triangles = o3d.utility.Vector3iVector(tri.simplices);
    return surface;

def get_volume(points):
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

    return volume;




def PlatRaycast_For_BraMesh(mesh,center=[0,0,0],direct=[0,0,1], resolution = 0.1,range_x = [-1.0,1.0],range_y = [-1.0,1.0]):
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



def visual_pointcloud(point_cloud_list=None, other_point_list=None, other_TM=None):
            
    point_mesh_list = [];

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate.scale(0.1, center=(0, 0, 0)); 
    point_mesh_list.append(coordinate);

    if(point_cloud_list!=None):
        for point_cloud_ in point_cloud_list:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_stream=point_cloud_.reshape(-1,3);
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd_stream);
            point_mesh_list.append(pcd_o3d);

    if(other_point_list!=None):
        for point_ in other_point_list:
            point_mesh = o3d.geometry.TriangleMesh.create_sphere()
            point_mesh.scale(0.005, center=(0, 0, 0));
            point_mesh.translate(point_);
            point_mesh.paint_uniform_color([0, 0, 1])
            point_mesh_list.append(point_mesh);

    if(other_TM!=None):
        other_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        other_coordinate.scale(0.1, center=(0, 0, 0));
        other_coordinate.transform(other_TM);
        point_mesh_list.append(other_coordinate);
    o3d.visualization.draw_geometries(point_mesh_list, mesh_show_back_face=True)

def reduce_mask_circle(mask,label_array,reduce_value = 3):
    # 이 함수는 지정된 label의 mask를 원형 기준으로 줄이는 기능을 한다.
    # 원형의 중심점을 mask 이미지에서 label의 평균 위치다.

    temp_mask = mask.copy();
    background_mask = np.zeros(temp_mask.shape,dtype=np.uint8)
    for label_number in label_array:
        mask_index= np.where(temp_mask==label_number);
        if(len(mask_index[0]) <= 0):
            continue;
        # 가로,세로 인덱스의 중심위치 계산함.
        center_y_idx=int(mask_index[0].mean());
        center_x_idx=int(mask_index[1].mean());

        label_idxs = np.vstack(mask_index[:2]).T;
        max_range = np.linalg.norm(label_idxs - [center_y_idx,center_x_idx],axis=1).max() + 1.0;
        
        
        
        theta_range = np.arange(0.0,2*np.pi,np.deg2rad(0.5));
        r_range = np.arange(0.0,max_range,1.0);

        for theta_ in theta_range:
            
            idx_list = [];
            for r_ in r_range:
                x_idx = center_x_idx + int(r_*np.cos(theta_));
                y_idx = center_y_idx + int(r_*np.sin(theta_));

                if(temp_mask[y_idx,x_idx][0]==label_number):
                    idx_list.append([y_idx,x_idx])

            for y_idx,x_idx in idx_list[:-reduce_value]:
                background_mask[y_idx,x_idx] = label_number;


    # cv2.imshow("mask",temp_mask)
    # cv2.imshow("reduce_mask",background_mask);
    # cv2.waitKey(0);
    return background_mask;


def fast_reduce_mask(mask,label_array,reduce_value = 3):
    # 이 함수는 지정된 label의 mask를 기준으로 줄이는 기능을 한다.(개선사항:속도, O(n^3) -> O(n));
    # 기준 중심점을 mask 이미지에서 label의 평균 위치다.

    temp_mask = mask.copy();
    h_,w_,c_ = temp_mask.shape;
    background_mask = np.zeros((h_,w_,c_),dtype=np.uint8)

    for label_number in label_array:
        mask_index = np.where(temp_mask==label_number);
        if(len(mask_index[0]) <= 0):
            continue;
        

        # 가로,세로 라벨의 중심위치 계산함.
        center_y_idx=int(mask_index[0].mean());
        center_x_idx=int(mask_index[1].mean());
        
        #라벨의 최외각 위치 계산
        min_y_idx = mask_index[0].min();
        max_y_idx = mask_index[0].max();
        min_x_idx = mask_index[1].min();
        max_x_idx = mask_index[1].max();

        #라벨 축소 작업f
        range_y_idx = max_y_idx-min_y_idx;
        range_x_idx = max_x_idx-min_x_idx;

        reduce_range_y_idx = range_y_idx - int(reduce_value*2);
        reduce_range_x_idx = range_x_idx - int(reduce_value*2);

        #선형 축소 방정식 계산
        reduce_mask_index_y = np.int64((mask_index[0]-center_y_idx)*(reduce_range_y_idx/range_y_idx)) + center_y_idx;
        reduce_mask_index_x = np.int64((mask_index[1]-center_x_idx)*(reduce_range_x_idx/range_x_idx)) + center_x_idx;

        #축소된 라벨 붙여 넣기 
        background_mask[(reduce_mask_index_y,reduce_mask_index_x,mask_index[2])] = temp_mask[mask_index];

    return background_mask;




def getPlateProject(points):
    
    #point_cloud dimension => C,X,Y,Z
    # points = [ [X1,Y1,Z1] , [X2,Y2,Z2] , [X3,Y3,Z3]]
    p1=np.array(points[0]);
    p2=np.array(points[1]);
    p3=np.array(points[2]);


    r1 = p2 - p1;
    r2 = p3 - p1;


    Normal_Vector = np.cross(r1, r2);
    Normal_UnitVector = Normal_Vector/np.linalg.norm(Normal_Vector);
    

    Plate_UnitVector1 = r1/np.linalg.norm(r1);
    Plate_UnitVector2 = np.cross(Normal_UnitVector, Plate_UnitVector1);


    return Plate_UnitVector1,Plate_UnitVector2,-Normal_UnitVector;



def Depth2PointCloud(depth_uint16,scale=0.001):
    # realsense 카메라(안티그래비티) Depth 이미지를 PC로 변환하는 함수
    
    fx = 607.62634277;
    fy = 607.40606689;
    cx = 641.01501465;
    cy = 366.62744141;
    h,w = depth_uint16.shape[:2];
    # camera_intrinsics = [[fx,0,cx],
    #                      [0,fy,cy],
    #                      [0, 0, 1]];

    pointcloud_xyz=np.zeros((h,w,3),dtype=np.float32);

    for h_idx,h_ in enumerate(depth_uint16):
        for w_idx,w_ in enumerate(h_):
            Z = scale * w_;
            X = ((w_idx - cx)/fx) * Z;
            Y = ((h_idx - cy)/fy) * Z;
            pointcloud_xyz[h_idx,w_idx] =   [X , Y, Z ];

    return pointcloud_xyz;


def improve_Depth2PointCloud(depth_uint16,scale=0.001):
    # realsense 카메라(안티그래비티) Depth 이미지를 PC로 변환하는 함수
    
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



def PointCloud2Depth(point_cloud,size=(0,0),scale=0.001):
    # realsense 카메라(안티그래비티) Depth 이미지를 PC로 변환하는 함수
    
    fx = 607.62634277;
    fy = 607.40606689;
    cx = 641.01501465;
    cy = 366.62744141;
    h,w = size;

    K = np.array([  [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]);

    depth_img=np.zeros((h,w),dtype=np.float32);
    point_cloud_stream = point_cloud.reshape(-1,3);

    for pc_xyz_ in point_cloud_stream:
        x,y,z=pc_xyz_;
        X = np.array([x, y, z]).T;
        if z > 0.1:
            uv_norm = X / z;
            uv = np.matmul(K, uv_norm)
            uv = uv.astype(np.int32)
            w_idx = uv[0];
            h_idx = uv[1];
            
            
            if((h<=h_idx) or (h_idx< 0)):
                continue;
            if((w<=w_idx) or (w_idx < 0)):
                continue;
            
            depth_img[h_idx,w_idx] = (z/scale);


    return depth_img;


def PointCloud2DepthPosition(point,scale=0.001):
    # realsense 카메라(안티그래비티) Depth 이미지를 PC로 변환하는 함수
    
    fx = 607.62634277;
    fy = 607.40606689;
    cx = 641.01501465;
    cy = 366.62744141;
    K = np.array([  [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]);
    x,y,z = point;
    X = np.array([x, y, z]).T;
    uv_norm = X /z
    uv = np.matmul(K, uv_norm)
    uv = uv.astype(np.int32)
    w_idx = uv[0];
    h_idx = uv[1];

    return (w_idx,h_idx);


def bra_fitting_func(fit_data_xy,limit_x=(-4,4),limit_y=(0,4),resolution = 1.0,func_name = "cubic", avg_Parameters = [], avg_weight = 0.0, custom_weight = None):
    # writer : dkko
    # date   : 20230119
    # e-mail : dkko@snuailab.ai

    # limit_x    = (-maximum_data_x, last_data_x )
    # limit_y    = (plate_offset ,bra_max_height)
    # resolution = depth image pixel size(defualt = 1)
    # func_name  = fitting function name

    x=fit_data_xy[0];
    y=fit_data_xy[1];
    bounds=(0,1)
    # 함수
    if(func_name=="linear"):
        def func(x, a, b): 
            return a*(x) + b
        initialParameters = np.array([1.0, 1.0 ]);

    if(func_name=="bra_quadratic"):
        def func(x, a, b, c ): 
            return -(((x-b)/a)**2) + c
        initialParameters = np.array([1.0, 1.0 ,1.0]);
        bounds=(-10000,10000)

    if(func_name=="sin"):
        def func(x, a, b ): 
            return np.sin(a*x) + b;
        initialParameters = np.array([1.0, 1.0]);

    if(func_name=="bra"):
        def func(x, a, b, c ,d): 
            return a*(x*x*x) + b*(x*x) + c*(x) + d;
        initialParameters = np.array([-0.1,-.1,-.1,.1]);
        bounds=( [-0.1,-.1,-.1,-0.1], [0, .1, 0.1, 0.1] )


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
    
    if(func_name=="circle"):
        def func(x, cx, cy, r):
            return np.sqrt( (r*r)- ((x-cx)*(x-cx)) ) + cy;
        initialParameters = np.array([1.0, 1.0 ,1.0]);
        def circfit(xs, ys):
            #원을 피팅하는 함수(개인 제작)
            a = matrix([[1,x,y,] for x,y in zip(xs, ys)])
            b = matrix([[(x*x + y*y)] for x,y in zip(xs, ys)])
            c = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(b)#inverse
            xc = 0.5 * c[1]
            yc = 0.5 * c[2]
            r = xc*xc + yc*yc + c[0];
            return xc,yc,np.sqrt(r);

    # 피팅
    fittedParameters, pcov = scipy.optimize.curve_fit(func, x, y, initialParameters, bounds=bounds);

    # def func(x, cx, cy, r):
    #     return np.sqrt( (r*r)- ((x-cx)*(x-cx)) ) + cy;
    
    
    
    # x_max = x.max();
    # x_min = x.min();
    # x_norm = (x - x_min)/(x_max-x_min);

    # y_max = y.max();
    # y_min = y.min();
    # y_norm = (y - y_min)/(y_max-y_min);

    # plt.plot(x_norm,y_norm);
    # plt.show();

    
    # xc,yc,r = circfit(list(x_norm),list(y_norm));
    # xc = np.array(xc);
    # yc = np.array(yc);
    # r = np.array(r);
    
    # if((func_name=="bra") and (fittedParameters[0]>0.0)):
    #     fittedParameters[0]=0.0;
        
    modelPredictions = func(x, *fittedParameters) 
    MeanAbsError = np.abs(modelPredictions - y).mean();
    Correlation_Coefficient=np.corrcoef(modelPredictions, y)[0, 1];
    
    # x 데이터 생성
    x=np.arange(limit_x[0], limit_x[1], resolution);
    y_hat = func(x, * fittedParameters );

    
    # if (len(avg_Parameters)!=0):
    #     avg_Parameters_ = avg_Parameters.copy();
    #     if(custom_weight!=None):
    #         avg_Parameters_[2] += custom_weight
    #     # 현재 데이터 모델 + 평균 모델 가중
    #     fittedParameters_=(( np.multiply((1.0-np.array(avg_weight)),fittedParameters)) + (np.multiply(np.array(avg_weight),avg_Parameters_) ) );
    #     y_hat = func(x, * fittedParameters_ );
    # # 현재 데이터 모델
    # else:
    #     y_hat = func(x, *fittedParameters);
    # x_max = x.max();
    # x_min = x.min();
    # x_norm = (x - x_min)/(x_max-x_min);

    # y_hat = func(x_norm, xc,yc,r);

    # y_hat =  (y_hat*(y_max-y_min))+y_min;
    # y_hat = np.array(y_hat)[0];
    
    # # plt.subplots(2, 1)
    # # plt.plot(x,y_hat);
    # # plt.show()


    y_limit=np.argwhere((y_hat<limit_y[1]) & (y_hat>limit_y[0]) );
    return x[y_limit], y_hat[y_limit],[MeanAbsError,Correlation_Coefficient];



def bra_suf_fitting_func(fit_data_xyz,more_xy=[],resolution = 0.001):
    # writer : dkko
    # date   : 20230215
    # e-mail : dkko@snuailab.ai

    # limit_x    = (-maximum_data_x, last_data_x )
    # limit_y    = (plate_offset ,bra_max_height)
    # resolution = depth image pixel size(defualt = 1)
    # func_name  = fitting function name

    x = fit_data_xyz[0];
    y = fit_data_xyz[1];
    z = fit_data_xyz[2];
    
    x_min = x.min();
    x_max = x.max();

    y_min = y.min();
    y_max = y.max();

    # 함수
    def func(data, a, b, c ,d, e):
        x_,y_ = data
        return -(((x_-b)/a)**2+((y_-d)/c)**2) + e
        
    initialParameters = np.array([1,1, 1, 1, 0.3]);

    # 피팅
    fittedParameters, pcov = scipy.optimize.curve_fit(func, np.vstack([x, y]), z , initialParameters);

        
    modelPredictions = func(np.vstack([x, y]), *fittedParameters) 
    MeanAbsError = np.abs(modelPredictions - z).mean();
    Correlation_Coefficient=np.corrcoef(modelPredictions, z)[0, 1];
    
    z_hat = func(np.vstack([more_xy[:,0], more_xy[:,1]]), * fittedParameters );

    return [more_xy[:,0],more_xy[:,1],z_hat],[MeanAbsError,Correlation_Coefficient];   



def get_masked_centerpoint(mask,lebel,show=False):
    # mask map에서 세그먼트의 중심 위치를 추출하는 기능 함수.
    
    # lebel과 일치하는 조건을 가진 가로세로 인덱스들을 추출함
    mask_index= np.where(mask==lebel);

    # 가로,세로 인덱스의 중심위치 계산함.
    mean_y_idx=int(mask_index[0].mean());
    mean_x_idx=int(mask_index[1].mean());
     
    # mask 추출한 위치에 이미지화
    if(show):
        mask[mean_y_idx-10:mean_y_idx+10,mean_x_idx-10:mean_x_idx+10] = 255;
        cv2.imshow("mask_img",mask);
        cv2.waitKey(0);
    return (mean_x_idx,mean_y_idx);


def get_near_centerpoint(depth, pos = [0,0],research_range = 20):
    # 깊이 이미지와 지정된 중심 위치로 부터 가장 가까운 값을 구하는 기능 함수.
    cp_x_idx,cp_y_idx=pos;
    
    if(depth[cp_y_idx,cp_x_idx]==0.0):
        r_range = research_range;#pixel
        
        temp_depth = depth.copy();

        range_value=temp_depth[cp_y_idx-r_range:cp_y_idx+r_range+1,cp_x_idx-r_range:cp_x_idx+r_range+1];
        range_value[np.where(range_value == 0.0)]= 65535.0;

        minmum_dy_idx, minmum_dx_idx = np.where(range_value == np.min(range_value))
        
        near_x_idx=cp_x_idx+(minmum_dx_idx[int(len(minmum_dx_idx)/2)]-r_range);
        near_y_idx=cp_y_idx+(minmum_dy_idx[int(len(minmum_dy_idx)/2)]-r_range);
    else:
        near_x_idx = cp_x_idx
        near_y_idx = cp_y_idx

    return (near_x_idx,near_y_idx);


def RGB2Depth(rgb,gain=1000.0):
    r,g,b = rgb
    return (((r + (g*(2**8)))+(b*(2**16) )) / (2**24)) * gain

def Depth2RGB(depth,gain=1000.0):
    temp_depth=(depth/gain)*(2**24);
    b = (int(temp_depth)>>16)&0xFF;
    g = (int(temp_depth)>>8)&0xFF;
    r = (int(temp_depth)>>0)&0xFF;

    return [r,g,b];
def BGR2Depth(rgb,gain=1000.0):
    b,g,r = rgb
    return (((r + (g*(2**8)))+(b*(2**16) )) / (2**24)) * gain
    

def Depth2BGR(depth,gain=1000.0):
    temp_depth=(depth/gain)*(2**24);
    b = (int(temp_depth)>>16)&0xFF;
    g = (int(temp_depth)>>8)&0xFF;
    r = (int(temp_depth)>>0)&0xFF;
    #return cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.3), cv2.COLORMAP_BONE);
    return [b,g,r];


def DepthInt16ToBGR(depth):
    h,w = depth.shape[:2];
    temp_depth_uint16 = depth.copy();
    temp_depth_rgb = np.zeros((h,w,3),dtype=np.uint8)
    
    for h_idx, h_ in enumerate(temp_depth_uint16):
        for w_idx, w_ in enumerate(h_):
            temp_depth_rgb[h_idx,w_idx,0] = (int(temp_depth_uint16[h_idx,w_idx])>>16)&0xFF;
            temp_depth_rgb[h_idx,w_idx,1] = (int(temp_depth_uint16[h_idx,w_idx])>>8)&0xFF;
            temp_depth_rgb[h_idx,w_idx,2] = (int(temp_depth_uint16[h_idx,w_idx])>>0)&0xFF;
    
    return temp_depth_rgb;


def improve_interpolate(depth, p1,p2,p3,cp, r_resulotion = 1.0,theta_resulotion = 1.0,depth_type = 'uint16', all = False):
    # r_resulotion demension is pixel
    # theta_resulotion demension is degree
    # p1,p2,p3,cp are center-point 
    if(depth_type == 'uint16'):
        depth_resolution = 2**16;
    if(depth_type == 'uint32'):
        depth_resolution = 2**32;
    cpx,cpy = cp;
    p1x,p1y = p1;
    p2x,p2y = p2;
    p3x,p3y = p3;

    # 속옷 중심점과 각 랜드마크의 각도(rad) 계산
    theta1= np.arctan2(( p1y - cpy) , (p1x - cpx));
    theta2= np.arctan2(( p2y - cpy) , (p2x - cpx));
    theta3= np.arctan2(( p3y - cpy) , (p3x - cpx));

    # 피팅할 원의 반지름 거리 계산
    radius1=np.linalg.norm(np.array(cp)-np.array(p1));
    radius2=np.linalg.norm(np.array(cp)-np.array(p2));
    radius3=np.linalg.norm(np.array(cp)-np.array(p3));


    radius=np.array([radius1,radius2,radius3]).max();
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
        theta_range = list(np.arange(0 ,np.pi*2, (np.pi/180.0)*1.0));
    
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
        if((len(temp_list_x_check)<10) or (len(temp_list_y_check)<10)):
            continue;

        x_hat,y_hat,parm=bra_fitting_func(  np.array([temp_list_x_check,temp_list_y_check]),
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
    return interpolation_depth,avg_parm,success_fit


def fast_interpolate_3d(depth_xyz, p1,p2,p3,cp, r_resulotion = 1.0,theta_resulotion = 1.0,depth_type = 'uint16', all = False):
    # 위 함수는 기존 interpolate의 속도를 개선한 함수이며, 정확도를 향샹시켰다.
    # r_resulotion demension is pixel
    # theta_resulotion demension is degree
    # p1,p2,p3,cp are center-point
    h,w=depth_xyz.shape[:2];
    if(depth_type == 'uint16'):
        depth_resolution = 2**16;
    if(depth_type == 'uint32'):
        depth_resolution = 2**32;
    cpx,cpy = cp;
    p1x,p1y = p1;
    p2x,p2y = p2;
    p3x,p3y = p3;

    # 속옷 중심점과 각 랜드마크의 각도(rad) 계산
    theta1= np.arctan2(( p1y - cpy) , (p1x - cpx));
    theta2= np.arctan2(( p2y - cpy) , (p2x - cpx));
    theta3= np.arctan2(( p3y - cpy) , (p3x - cpx));

    # 피팅할 원의 반지름 거리 계산
    radius1=np.linalg.norm(np.array(cp)-np.array(p1));
    radius2=np.linalg.norm(np.array(cp)-np.array(p2));
    radius3=np.linalg.norm(np.array(cp)-np.array(p3));
    radius=np.array([radius1,radius2,radius3]).max();
    # print("radius:",radius)
    # print(p1x , cpx);


    # pixel 별 반지름 데이터 생성
    r_range = np.linspace(0.0,radius,num=int(radius/r_resulotion));
    more_r_range = np.linspace(radius,radius+200,num=200);

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
        theta_range = list(np.arange(0 ,np.pi*2, (np.pi/180.0)*1.0));
    
    # print("theta_range",theta_range, )
    # print("theta1 : ",np.rad2deg(theta1));
    # print("theta2 : ",np.rad2deg(theta2));
    # print("theta3 : ",np.rad2deg(theta3));    
    
        
    # 연장 부분의 평균 모델의 형상 변수
    avg_parm = [];

    # 연장한 부분을 저장할 depth 이미지 변수
    interpolation_depth = np.zeros((depth_xyz.shape));
    success_fit = True;
    h_pad = 5;



    
    temp_list_xyz = [];
    more_xy = [];
    count = 0;
    merge_xyz_hat = [];
    # cp위치로 부터 반지름 크기와 방향으로 x,y 좌표 계산하여 피팅함수 계산
    for theta_ in theta_range:
        
        r_data = []
        is_first=True;
        pre_x = 0;
        pre_y = 0;

        sum_dxdr = 0;
        sum_dydr = 0;
        count_dr = 0;
        last_x = 0;
        last_y = 0;


        for r_ in r_range:
            temp_x = cpx + int(r_ *np.cos(theta_)) ;
            temp_y = cpy + int(r_ *np.sin(theta_)) ;
            x_,y_,z_=depth_xyz[temp_y,temp_x];
            if(np.isnan(x_) or np.isnan(y_) or np.isnan(z_)):
                continue;
            temp_list_xyz.append([x_,y_,z_]);
            
            if(is_first):
                is_first=False;
                pre_x = x_;
                pre_y = y_;
                last_x = x_;
                last_y = y_;
                continue;

            dxdr=(x_-pre_x)/r_resulotion;
            dydr=(y_-pre_y)/r_resulotion;
            pre_x = x_;
            pre_y = y_;
            sum_dxdr+=dxdr;
            sum_dydr+=dydr;
            count_dr+=1;
            last_x = x_;
            last_y = y_;

        avg_dxdr = sum_dxdr/count_dr;
        avg_dydr = sum_dydr/count_dr;

        for r_idx,r_ in enumerate(more_r_range):
            pred_x=(avg_dxdr*r_idx*r_resulotion) + last_x;
            pred_y=(avg_dydr*r_idx*r_resulotion) + last_y;
            more_xy.append([pred_x,pred_y])

        count+=1;
        if(count%60==50):
            more_xy = np.array(more_xy);
            temp_list_xyz=np.array(temp_list_xyz);
            xyz_,parm=bra_suf_fitting_func(  np.array([temp_list_xyz[:,0],temp_list_xyz[:,1],temp_list_xyz[:,2]]),
                                                    more_xy=more_xy,);

            xyz_hat = np.concatenate([np.expand_dims(xyz_[0],1) ,np.expand_dims(xyz_[1],1) ,np.expand_dims(xyz_[2],1) ],axis=1);
            merge_xyz_hat.extend(xyz_hat)
            temp_list_xyz = [];
            more_xy = [];
    # print(xyz_)
    # pcd_hat = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_hat));
    # pcd_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(temp_list_xyz));
    # pcd_ref.paint_uniform_color([0., 1., 0.])
    # o3d.visualization.draw_geometries([pcd_hat,pcd_ref], mesh_show_back_face=True);
    # exit();
    xyz_hat_depth=PointCloud2Depth(np.array(merge_xyz_hat),(h,w));
    return xyz_hat_depth,parm,True


         

    



def get_masked_depth(depth,mask,lebel_list):
    # depth 이미지에서 라벨_리스트의 정보만 추츨

    depth_dst=np.zeros(depth.shape);

    for lebel in lebel_list:
        mask_inex = np.where(mask==lebel);
        for (y,x) in zip(mask_inex[0],mask_inex[1]):
            depth_dst[y,x] = depth[y,x];
    
    return depth_dst;

def except_masked_depth(depth,mask,lebel_list):
    # depth 이미지에서 라벨_리스트의 정보만 제외
    temp_depth_img = depth.copy();
    for lebel in lebel_list:
        mask_inex = np.where(mask==lebel);
        for (y,x) in zip(mask_inex[0],mask_inex[1]):
            temp_depth_img[y,x] = 0;
    return temp_depth_img;


def check_anti_data(color_img, depth_img, mask_img, is_show = False):
    # 이 함수는 안티그래비티 뎁스 이미지와 마스크(mask)이미지 검수 및 시각적 확인을 한다.

    
    if(is_show):
        masked_pcd = Depth2PointCloud(depth_img,0.001)

        lp1=masked_pcd[bra_l_1_point[1],bra_l_1_point[0]]
        lp2=masked_pcd[bra_l_2_point[1],bra_l_2_point[0]]
        lp3=masked_pcd[bra_l_3_point[1],bra_l_3_point[0]]
        lpcp=masked_pcd[bra_l_cp[1],bra_l_cp[0]]

        land1_mesh = o3d.geometry.TriangleMesh.create_sphere()
        land2_mesh = o3d.geometry.TriangleMesh.create_sphere()
        land3_mesh = o3d.geometry.TriangleMesh.create_sphere()
        land1_mesh.scale(0.005, center=(0, 0, 0));
        land2_mesh.scale(0.005, center=(0, 0, 0));
        land3_mesh.scale(0.005, center=(0, 0, 0));
        land1_mesh.translate(lp1);
        land2_mesh.translate(lp2);
        land3_mesh.translate(lp3);
        land1_mesh.paint_uniform_color([0, 0, 1])
        land2_mesh.paint_uniform_color([0, 0, 1])
        land3_mesh.paint_uniform_color([0, 0, 1])

        
        origin_pcd_o3d = o3d.geometry.PointCloud()
        origin_pcd_stream=Depth2PointCloud(depth_img,0.001).reshape(-1,3);
        origin_pcd_o3d.points = o3d.utility.Vector3dVector(origin_pcd_stream)

        masked_pcd_o3d = o3d.geometry.PointCloud()
        masked_pcd_stream=masked_pcd.reshape(-1,3);
        masked_pcd_stream_idx = np.where(masked_pcd_stream[:,2]>0.1);
        masked_pcd_stream=masked_pcd_stream[masked_pcd_stream_idx];
        masked_pcd_o3d.points = o3d.utility.Vector3dVector(masked_pcd_stream)
        color_stream=color_img.reshape(-1,3)[masked_pcd_stream_idx][:,[2,1,0]];
        masked_pcd_o3d.colors = o3d.utility.Vector3dVector(color_stream/255.0);

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        coordinate.scale(0.1, center=(0, 0, 0)); 

        print("lelf_P1-lelf_P2 distance : ", np.linalg.norm(lp3-lp1)," m");
        #o3d.visualization.draw_geometries([origin_pcd_o3d,masked_pcd_o3d,coordinate,land1_mesh,land2_mesh,land3_mesh], mesh_show_back_face=True)

        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #     labels = np.array(masked_pcd_o3d.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        # max_label = labels.max()
        # erter_masked_pcd_o3d = o3d.geometry.PointCloud();
        # np_masked_pcd_o3d = np.array(masked_pcd_o3d.points);
        # erter_masked_pcd_o3d.points = np_masked_pcd_o3d[np.where(labels<0)];
            
        
        
        # print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # masked_pcd_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])
        

        o3d.visualization.draw_geometries([masked_pcd_o3d,coordinate,land1_mesh,land2_mesh,land3_mesh], mesh_show_back_face=True)

    

if __name__ == "__main__":
    # maker_name = "02. 도로시와_츠메르_볼륨메이커"
    # bra_cat = "dorosiwa_B_10001_70_B"

    bra_cat = "shoomeg_75A"
    
    num = "03";
    root = "/home/ljj/data/anti/test"
    #root = "/media/dkko/datadrive1/안티그래비티_자료들/5_종진님_/anti102"
    
    color_path=root+"/images/"+bra_cat+"_img_"+num+".png";
    depth_path=root+"/npy/"+bra_cat+"_depth_"+num+".npy";
    mask_path=root+"/segmentations/"+bra_cat+"_mask_"+num+".png";
    
    
    # depth_path="/media/dkko/DATADRIVE1/안티그래비티_자료들/5_종진님_/test_data/eval_depth_img.npy";
    # mask_path="/media/dkko/DATADRIVE1/안티그래비티_자료들/5_종진님_/test_data/eval_mask_img.png";
    


    # color_path="/home/dkko/Downloads/PAD/"+maker_name+"/images/"+bra_cat+"_img_04.png";
    # depth_path="/home/dkko/Downloads/PAD/"+maker_name+"/npy/"+bra_cat+"_depth_04.npy";
    # mask_path="/home/dkko/Downloads/PAD/"+maker_name+"/segmentations/"+bra_cat+"_mask_04.png";
    # pick_path = "/home/dkko/Downloads/PAD/01. 도로시와_노와이어_풀샷브라/pickle/dorosiwa_A_10001_70_A_00.pickle";


    depth_img=np.load(depth_path);

    origin_depth_img = depth_img.copy();


    # cv2.IMREAD_UNCHANGED : 이미지를 로드할때 비트수를 변환하지 않고 원본 그대로 로드
    color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED);    
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED);
    # cv2.imshow("color_img",color_img);
    # cv2.imshow("mask_img",mask_img);
    # cv2.imshow("depth_img",DepthInt16ToBGR(depth_img));
    # cv2.waitKey(0);

    
    reduce_mask_range = 10;#라벨을 줄이는 반지름 크기

    # bra_l_1_label = 29
    # bra_l_2_label = 5
    # bra_l_3_label = 14
    # bra_l_label = 33
    # brawire_l_label = 11


    # bra_l_1_label = 29
    # bra_l_2_label = 5
    # bra_l_3_label = 32
    # bra_l_label = 33
    # brawire_l_label = 11

    bra_l_1_label = 29
    bra_l_2_label = 5
    bra_l_3_label = 32
    bra_l_label = 33
    brawire_l_label = 11


    bra_r_1_label = 13
    bra_r_2_label = 6
    bra_r_3_label = 35
    bra_r_label = 15
    brawire_r_label = 28

    bra_l_1_point = get_masked_centerpoint(mask_img,bra_l_1_label);
    bra_l_2_point = get_masked_centerpoint(mask_img,bra_l_2_label);
    bra_l_3_point = get_masked_centerpoint(mask_img,bra_l_3_label);
    bra_l_cp = get_masked_centerpoint(mask_img,bra_l_label);


    masked_depth = get_masked_depth( depth_img,
                                    fast_reduce_mask(mask_img,[bra_l_label,brawire_l_label],reduce_mask_range),
                                     [bra_l_label,brawire_l_label]);
    bra_l_1_point=get_near_centerpoint(masked_depth,pos=bra_l_1_point,research_range=reduce_mask_range*2);
    bra_l_2_point=get_near_centerpoint(masked_depth,pos=bra_l_2_point,research_range=reduce_mask_range*2);
    bra_l_3_point=get_near_centerpoint(masked_depth,pos=bra_l_3_point,research_range=reduce_mask_range*2);
    
    #데이터 검사
    import time
    s=time.time();
    interpo_depth,validation,res = improve_interpolate(masked_depth,bra_l_1_point,bra_l_2_point,bra_l_3_point,bra_l_cp);
    print("improve_interpolate: ",time.time()-s);
    s=time.time();
    #interpo_depth,validation,res = fast_interpolate_3d(improve_Depth2PointCloud(np.uint16(masked_depth)),bra_l_1_point,bra_l_2_point,bra_l_3_point,bra_l_cp);
    print("fast_interpolate_3d: ",time.time()-s);
    print("interpolation_validation",validation);

    # res = False;
    # if(res==False):
    #     print("interpolate faile : fsfsdfsdf");
    #     n,o,a = getPlateProject([lp1,lp2,lp3]);
    #     T_TM=np.array([ [n[0],n[1],n[2]],
    #                     [o[0],o[1],o[2]],
    #                     [a[0],a[1],a[2]],]).T;

    #     h,w = masked_depth.shape[:2];
    #     pcd_masked_depth = Depth2PointCloud(masked_depth,0.001);

    #     lp1=pcd_masked_depth[bra_l_1_point[1],bra_l_1_point[0]]
    #     lp2=pcd_masked_depth[bra_l_2_point[1],bra_l_2_point[0]]
    #     lp3=pcd_masked_depth[bra_l_3_point[1],bra_l_3_point[0]]
    #     lpcp=pcd_masked_depth[bra_l_cp[1],bra_l_cp[0]]

    #     temp_lp1 = lp1;

    #     t_pcd_masked_depth = np.matmul((pcd_masked_depth - temp_lp1),T_TM)+temp_lp1;
    #     t_lp1 = np.matmul((lp1-temp_lp1),T_TM)+temp_lp1;
    #     t_lp2 = np.matmul((lp2-temp_lp1),T_TM)+temp_lp1;
    #     t_lp3 = np.matmul((lp3-temp_lp1),T_TM)+temp_lp1;
    #     t_lpcp = np.matmul((lpcp-temp_lp1),T_TM)+temp_lp1;


    #     # t_pcd_masked_depth = np.matmul((t_pcd_masked_depth - temp_lp1),T_TM.T)+temp_lp1;
    #     # t_lp1 = np.matmul((t_lp1-temp_lp1),T_TM.T)+temp_lp1;
    #     # t_lp2 = np.matmul((t_lp2-temp_lp1),T_TM.T)+temp_lp1;
    #     # t_lp3 = np.matmul((t_lp3-temp_lp1),T_TM.T)+temp_lp1;
    #     # t_lpcp = np.matmul((t_lpcp-temp_lp1),T_TM.T)+temp_lp1;


    #     t_bra_l_1_point = PointCloud2DepthPosition(t_lp1)
    #     t_bra_l_2_point  = PointCloud2DepthPosition(t_lp2)
    #     t_bra_l_3_point  = PointCloud2DepthPosition(t_lp3)
    #     t_bra_l_cp = PointCloud2DepthPosition(t_lpcp)

    #     sdfsdf_t_pcd_masked_depth =PointCloud2Depth(t_pcd_masked_depth,(h,w));

    #     sdfsdf_t_pcd_masked_depth[t_bra_l_1_point[1],t_bra_l_1_point[0]] = 255;
    #     sdfsdf_t_pcd_masked_depth[t_bra_l_2_point[1],t_bra_l_2_point[0]] = 255;
    #     sdfsdf_t_pcd_masked_depth[t_bra_l_3_point[1],t_bra_l_3_point[0]] = 255;

    #     cv2.imshow("masked_depth",masked_depth);
    #     cv2.imshow("sdfsdf_t_pcd_masked_depth",DepthInt16ToBGR(sdfsdf_t_pcd_masked_depth));
    #     cv2.waitKey(0);
    #     interpo_depth,res = interpolate(sdfsdf_t_pcd_masked_depth,t_bra_l_1_point,t_bra_l_2_point,t_bra_l_3_point,t_bra_l_cp);
        
        
        
    #     visual_pointcloud([t_pcd_masked_depth,Depth2PointCloud(interpo_depth,0.001)],other_point_list=[t_lp1,t_lp2,t_lp3]);
        
    #     pcd_interpo_depth = Depth2PointCloud(interpo_depth,0.001);
    #     t_pcd_interpo_depth = np.matmul((pcd_interpo_depth - temp_lp1),T_TM.T)+temp_lp1;
    #     interpo_depth =PointCloud2Depth(t_pcd_interpo_depth,(h,w));

    #     cv2.imshow("interpo_depth",DepthInt16ToBGR(interpo_depth));
    #     cv2.waitKey(0);
        
        #exit();


    interpo_depth = except_masked_depth(    interpo_depth,
                                            fast_reduce_mask(mask_img,[bra_l_label,brawire_l_label],reduce_mask_range+2),
                                            [bra_l_label,brawire_l_label]);
    

    # 이 아래부터는 계산결과를 비주얼화하는 과정임.
    depth_img=np.zeros((masked_depth.shape[0],masked_depth.shape[1],3),dtype=np.float32);
    interpo_depth_img=np.zeros((interpo_depth.shape[0],interpo_depth.shape[1],3),dtype=np.float32);
    

    
    interpo_depth_img = improve_Depth2PointCloud(np.uint16(interpo_depth),0.001)
    masked_depth_img = improve_Depth2PointCloud(np.uint16(masked_depth),0.001)
    depth_pcd = improve_Depth2PointCloud(origin_depth_img,0.001)


    lp1=masked_depth_img[bra_l_1_point[1],bra_l_1_point[0]]
    lp2=masked_depth_img[bra_l_2_point[1],bra_l_2_point[0]]
    lp3=masked_depth_img[bra_l_3_point[1],bra_l_3_point[0]]
    lpcp=masked_depth_img[bra_l_cp[1],bra_l_cp[0]]



    
    print(lp1,lp2,lp3)

    n,o,a = getPlateProject([lp1,lp2,lp3]);
    normal_vector = a;
    p = lp1
    print("normal_vector",normal_vector, " z :" ,np.sqrt(normal_vector[0]*normal_vector[0] + normal_vector[1] * normal_vector[1]));
    

    #만약 평면을 찾을 수 없다면, 카메라 좌표계기준으로 최소점을 찾는 Transform Matrix 계산
    success_plate = True;
    if (np.sqrt(normal_vector[0]*normal_vector[0] + normal_vector[1] * normal_vector[1]) > 0.707):
        n = np.array([1.0, 0.0, 0.0]);
        o = np.array([0.0, -1.0, 0.0]);
        a = np.array([0.0, 0.0, -1.0]);
        normal_vector = a;

        minimum_z=masked_depth_img[np.where( np.linalg.norm(masked_depth_img,axis=2)> 0.1)][:,2].min();

        lp1[2]=minimum_z;

    
    
    
    plate_TM = o3d.geometry.TriangleMesh.create_coordinate_frame()
    plate_TM.scale(0.1, center=(0, 0, 0));
    plate_TM.rotate(np.array([n,o,a]).T, center=(0, 0, 0));
    plate_TM.translate(p);



    masked_depth_img=masked_depth_img[np.where( np.dot(masked_depth_img-p, normal_vector.T) > 0.0)];
    interpo_depth_img=interpo_depth_img[np.where( np.dot(interpo_depth_img-p, normal_vector.T) > 0.0)];
    



    pcd = o3d.geometry.PointCloud()
    depth_img_stream=masked_depth_img.reshape(-1,3);
    depth_img_stream=depth_img_stream[np.where(depth_img_stream[:,2]>0.1)];
    pcd.points = o3d.utility.Vector3dVector(depth_img_stream)

    # Point clould filtering 결과
    inlier_rad_pcd = o3d.geometry.PointCloud()
    inlier_rad_pcd.points = o3d.utility.Vector3dVector(depth_img_stream)


    interpo_pcd = o3d.geometry.PointCloud()
    interpo_depth_img_stream=interpo_depth_img.reshape(-1,3);
    interpo_depth_img_stream=interpo_depth_img_stream[np.where(interpo_depth_img_stream[:,2]>0.1)];
    interpo_pcd.points = o3d.utility.Vector3dVector(interpo_depth_img_stream)


    




    color_temp = np.zeros((interpo_depth_img_stream.shape))+[0,255,0]
    color_temp/=255.0;
    interpo_pcd.colors = o3d.utility.Vector3dVector(color_temp);

    
    for i in range(0):
        cl, ind = inlier_rad_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0);
        inlier_rad_pcd = inlier_rad_pcd.select_by_index(ind);
        # cl, ind = inlier_rad_pcd.remove_radius_outlier(nb_points=16, radius=0.005);
        # inlier_rad_pcd = inlier_rad_pcd.select_by_index(ind);
    
    
    asdasd = PointCloud2Depth(np.asarray(inlier_rad_pcd.points),size=(masked_depth.shape[0],masked_depth.shape[1]));
    # cv2.imshow("asdasd",DepthInt16ToBGR(asdasd));
    # cv2.waitKey(0);

    inlier_rad_pcd = inlier_rad_pcd.voxel_down_sample(voxel_size=0.001);
    interpo_pcd = interpo_pcd.voxel_down_sample(voxel_size=0.001);

    
    inlier_rad_pcd.translate(-p);
    inlier_rad_pcd.rotate(np.array([n,o,a]), center=(0, 0, 0));
    interpo_pcd.translate(-p);
    interpo_pcd.rotate(np.array([n,o,a]), center=(0, 0, 0));



    alpha = 0.005
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(inlier_rad_pcd+interpo_pcd, alpha)
    mesh.compute_vertex_normals()
    
    downpdc = mesh.sample_points_uniformly(number_of_points=5000)
    volume = get_volume(downpdc.points)

    #mesh=o3d.geometry.VoxelGrid.create_from_point_cloud(inlier_rad_pcd+interpo_pcd,voxel_size=0.002)

    points = PlatRaycast_For_BraMesh(   mesh = mesh,
                                        center = [0,0,-0.1],
                                        direct = [0,0,1],
                                        resolution = 0.001,
                                        range_x=[-0.5,0.5],
                                        range_y=[-0.5,0.5]);


    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));                            
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0);
    pcd = pcd.select_by_index(ind);


    
    volume2 = get_volume(pcd.points)

    print(f"The volume of pad(without ray) is: {round(volume, 7)} m3 -> "+str(volume*1e6)+"cc")
    print(f"The volume of pad(with ray) is: {round(volume2, 7)} m3 -> "+str(volume2*1e6)+"cc")


    depth_pcd_o3d=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_pcd.reshape(-1,3)));
    depth_pcd_o3d.colors = o3d.utility.Vector3dVector(color_img.reshape(-1,3)/255.0)

    
    # pcd = get_surface(pcd.points)
    pcd.rotate(np.array([n,o,a]).T, center=(0, 0, 0));
    pcd.translate(p);
    # pcd.paint_uniform_color([0., 1., 0.])

    inlier_rad_pcd.rotate(np.array([n,o,a]).T, center=(0, 0, 0));
    inlier_rad_pcd.translate(p);
    inlier_rad_pcd.paint_uniform_color([1., 0., 1.])

    land1_mesh = o3d.geometry.TriangleMesh.create_sphere()
    land2_mesh = o3d.geometry.TriangleMesh.create_sphere()
    land3_mesh = o3d.geometry.TriangleMesh.create_sphere()
    land1_mesh.scale(0.005, center=(0, 0, 0));
    land2_mesh.scale(0.005, center=(0, 0, 0));
    land3_mesh.scale(0.005, center=(0, 0, 0));
    land1_mesh.translate(lp1);
    land2_mesh.translate(lp2);
    land3_mesh.translate(lp3);


    o3d.visualization.draw_geometries([depth_pcd_o3d,pcd,inlier_rad_pcd,land1_mesh,land2_mesh,land3_mesh], mesh_show_back_face=True)



    # o3d.visualization.draw_geometries([inlier_rad_pcd,interpo_pcd])
    # inlier_rad_pcd.rotate(np.array([n,o,a]).T, center=(0, 0, 0));
    # inlier_rad_pcd.translate(p);
    # interpo_pcd.rotate(np.array([n,o,a]).T, center=(0, 0, 0));
    # interpo_pcd.translate(p);
    

    # alpha = 0.018
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(inlier_rad_pcd+interpo_pcd, alpha)
    # mesh.compute_vertex_normals()

    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # coordinate.scale(0.1, center=(0, 0, 0)); 
 

    # land1_mesh = o3d.geometry.TriangleMesh.create_sphere()
    # land2_mesh = o3d.geometry.TriangleMesh.create_sphere()
    # land3_mesh = o3d.geometry.TriangleMesh.create_sphere()
    # land1_mesh.scale(0.005, center=(0, 0, 0));
    # land2_mesh.scale(0.005, center=(0, 0, 0));
    # land3_mesh.scale(0.005, center=(0, 0, 0));
    # land1_mesh.translate(lp1);
    # land2_mesh.translate(lp2);
    # land3_mesh.translate(lp3);



    # o3d.visualization.draw_geometries([mesh,pcd,coordinate,plate_TM , land1_mesh,land2_mesh,land3_mesh], mesh_show_back_face=True)


