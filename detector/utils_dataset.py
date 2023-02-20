import numpy as np
import cv2
import os
import glob
import argparse
import numpy as np
import cv2
import os
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
import random
import shutil
import json
from skimage import measure
from shapely.geometry import Polygon

def get_pad(mask: np.ndarray, colors) -> np.ndarray:
    h, w, c = mask.shape
    pad_mask = np.zeros((h, w, c), dtype=np.uint8)
    for color in colors:
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color)*255, dtype=np.uint8)
        pad_mask += mask_copy

    return pad_mask

def get_landmark(mask: np.ndarray, colors: list) -> list:
    landmark_lst = []
    for i, color in enumerate(colors):
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color)*255, dtype=np.uint8)
        gray = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 1, param2 = 1, minRadius = 0, maxRadius = 50)
    
        landmark_lst += [int(circles[0][0][0]), int(circles[0][0][1]), 2]
        '''
        circle의 센터점 , (x, y, v)로 설정하는데 x, y는 2차원 평면상의 좌표이고 v는 visiblity이다.
        v=0; 레이블되지 않음(not labeled)
        v=1; 레이블링이 되어있지만 보이지 않음 (labeled but not visible)
        v=2; 레이블링이 되어 있고 볼 수 있음 (labeled and visible)
        우리의 데이터는 전부 2로 표기
        간혹 안보이는 점에 대해서 신중하게 할 경우 그 점에 대해서만 바꿔주어야함.
        '''
    return landmark_lst

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            pixel = mask_image.getpixel((x,y))[:3]

            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        if len(contour) > 30:
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            poly = Polygon(contour)
            
            if(poly.is_empty):
                continue

            polygons.append(poly)

            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
    
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "id": value,
            "name": key,
            "keypoints": list(range(3*3)),
            "skeleton" : []
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format_landmark_bra(boxes, image_id, category_id, annotation_id, keypoint):
    x1=min(list(boxes[:,0]) + list(boxes[:,2]))-10
    x2=max(list(boxes[:,0]) + list(boxes[:,2]))+10
    y1=min(list(boxes[:,1]) + list(boxes[:,3]))-5
    y2=max(list(boxes[:,1]) + list(boxes[:,3]))+5
    width =  x2 - x1
    height =  y2 - y1
    keypoint = sum(keypoint, [])
    boxes = np.array([x1, y1, width, height])
    area = (y2-y1) * (x2-x1)
    bbox = list(boxes)

    annotation = {
        'area': area,
        'bbox' : bbox,
        'category_id' : category_id,
        'id' : annotation_id,
        'image_id' : image_id,
        'iscrowd' : 0,
        'num_keypoints' : 9,
        'keypoints' : keypoint,
    }

    return annotation

def create_annotation_format_landmark_pad(polygon, image_id, category_id, annotation_id, keypoint):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        'area': area,
        'bbox' : bbox,
        'category_id' : category_id,
        'id' : annotation_id,
        'image_id' : image_id,
        'iscrowd' : 0,
        'num_keypoints' : 4,
        'keypoints' : keypoint,
    }

    return annotation

def create_annotation_format_mask(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        'area': area,
        'bbox' : bbox,
        'category_id' : category_id,
        'id' : annotation_id,
        'image_id' : image_id,
        'iscrowd' : 0,
        'num_keypoints' : 3,
        'segmentation' : segmentation
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [{}],
        "annotations": []
    }

    return coco_format

coco_config={
        "category_ids" : {
            "pad": 1
        },
        "category_colors" : {
            "(100, 100, 100)": 1
        },
        "colors" : {
            "pad_l": [33, 33, 33],
            "pad_r": [15,15,15],
            "wire_l": [11,11,11],
            "wire_r": [28,28,28],
            "landmark_t1": [26,26,26],
            "landmark_l1": [29,29,29],
            "landmark_b1": [5,5,5],
            "landmark_r1": [32, 32, 32],
            "landmark_c": [14,14,14],
            "landmark_l2": [35, 35, 35],
            "landmark_b2": [6,6,6],
            "landmark_r2": [13,13,13],
            "landmark_t2": [1,1,1],
        },
    }

class mask2coco(object):
    '''
    segmentations mask정보를 이용하여 coco format(json)으로 변환한다.
    활용도 측면에서 이 방식이 가장 깔끔함.
    dataset
    ├── annotations # Annotation 정보들 (file path, polygon, class 등등)
    │   ├── train_{name}.json
    │   ├── val_{name}.json
    │   └── test_{name}.json
    ├── train
    ├── val
    └── test
    '''
    def __init__(self, args, data_type):
        self.category_colors = coco_config["category_colors"]
        self.image_id = 0
        self.landmark_colors = []
        self.annotation_id = 0
        self.coco_format = get_coco_json_format()
        self.coco_format["categories"] = create_category_annotation(coco_config['category_ids'])
        self.dataset_path = os.path.join(args.data_root, '{}_{}'.format(args.dataset, args.type))
        self.save_path = os.path.join(self.dataset_path, 'annotations')
        self.mask_images = glob.glob(os.path.join(self.dataset_path, data_type, "segmentations/*"))
        self.file_name = data_type + ".json"
        self.dataset_type = args.type
        self.data_type = data_type 
        self.args = args

    def convert(self):
        for mask_image in tqdm(self.mask_images):
            # if not 'dorosiwa_B_10001_75_A_mask_05.png' in mask_image:
            #     continue
            image_filename = os.path.basename(mask_image).replace('mask','img')
            print(image_filename)
            mask_cv2=cv2.imread(mask_image)
            h, w = mask_cv2.shape[:2]

            image = create_image_annotation(image_filename, w, h, self.image_id)
            self.coco_format['images'].append(image)

            sub_mask1 = get_pad(mask_cv2, colors=[coco_config['colors']['pad_l'], 
                                                  coco_config['colors']['wire_l'], 
                                                  coco_config['colors']['landmark_t1'], 
                                                  coco_config['colors']['landmark_l1'], 
                                                  coco_config['colors']['landmark_b1'], 
                                                  coco_config['colors']['landmark_r1']])
            # key_points1 = get_landmark(mask_cv2, colors = [(5,5,5), (26,26,26), (29,29,29)])
            if self.args.key_type=='pad':
                key_points1 = get_landmark(mask_cv2, colors = [coco_config['colors']['landmark_t1'],
                                                            coco_config['colors']['landmark_l1'], 
                                                            coco_config['colors']['landmark_b1'],
                                                            coco_config['colors']['landmark_r1']])
            else:
                key_points1 = get_landmark(mask_cv2, colors = [coco_config['colors']['landmark_t1'],
                                                            coco_config['colors']['landmark_l1'], 
                                                            coco_config['colors']['landmark_b1'],
                                                            coco_config['colors']['landmark_r1'],
                                                            coco_config['colors']['landmark_c']])

            sub_mask2 = get_pad(mask_cv2, colors=[coco_config['colors']['pad_r'], 
                                                coco_config['colors']['wire_r'], 
                                                coco_config['colors']['landmark_l2'], 
                                                coco_config['colors']['landmark_b2'], 
                                                coco_config['colors']['landmark_r2'], 
                                                coco_config['colors']['landmark_t2']])

            key_points2 = get_landmark(mask_cv2, colors = [coco_config['colors']['landmark_t2'],
                                                           coco_config['colors']['landmark_r2'], 
                                                           coco_config['colors']['landmark_b2'], 
                                                           coco_config['colors']['landmark_l2']])
            gray1 = cv2.cvtColor(sub_mask1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(sub_mask2, cv2.COLOR_BGR2GRAY)
            sub_masks = {'(100, 100, 100)': [[gray1, key_points1], [gray2, key_points2]]}

            if self.dataset_type=='point' and self.args.key_type=='pad':
                self.pad_keypoints_convert(sub_masks)
            elif self.dataset_type=='point' and self.args.key_type=='bra':
                self.bra_keypoints_convert(sub_masks)
            elif self.dataset_type=='mask':
                self.mask_convert(sub_masks)
            self.image_id += 1

        save_path = os.path.join(self.save_path, self.file_name) 
        with open(save_path, "w") as outfile:
            json.dump(self.coco_format, outfile)
        print(f"Save {save_path}")

        return self.coco_format
    
        
    def pad_keypoints_convert(self, sub_masks):
        for color, masks in sub_masks.items():
            for sub_mask, key_points in masks:
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
                category_id = self.category_colors[color]
                for i in range(len(polygons)):
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    annotation = create_annotation_format_landmark_pad(polygons[i], self.image_id, category_id, self.annotation_id, key_points)
                    self.coco_format['annotations'].append(annotation)
                    self.annotation_id += 1

    
    def bra_keypoints_convert(self, sub_masks):
        boxes = []
        keypoints = []
        for color, masks in sub_masks.items():
            for sub_mask, key_points in masks:
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
                category_id = self.category_colors[color]
                boxes.append(polygons[0].bounds)
                keypoints.append(key_points)
            boxes = np.array(boxes)
            annotation = create_annotation_format_landmark_bra(boxes, self.image_id, category_id, self.annotation_id, keypoints)
            self.coco_format['annotations'].append(annotation)
            self.annotation_id +=1
            

    def mask_convert(self, sub_masks):
        for color, masks in sub_masks.items():
            for sub_mask, _ in masks:
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
                category_id = self.category_colors[color]
                for i in range(len(polygons)):
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    annotation = create_annotation_format_mask(polygons[i], segmentation, self.image_id, category_id, self.annotation_id)
                    self.coco_format['annotations'].append(annotation)
                    self.annotation_id += 1


    def visualize(self):
        annFile = os.path.join(self.save_path, self.file_name)
        coco = COCO(annFile)
        print(coco)
        catIds = [1]
        ids = coco.getImgIds(catIds=catIds)
        print(ids)
        for imgIds in ids[:3]:
            annIds = coco.getAnnIds(imgIds = imgIds, catIds=catIds)
            anns = coco.loadAnns(annIds)
            imgInfo = coco.loadImgs(imgIds)
            # print(anns)
            # print(self.dataset_path)
            image_name = os.path.join(self.dataset_path, self.data_type, 'images', imgInfo[0]['file_name'])
            image = Image.open(image_name).convert('RGB')
            print(image_name)
            plt.imshow(image)
            coco.showAnns(anns, draw_bbox=True)
            plt.show()
    



class split_dataset():
    '''
    azure_kinect_pad 
    ├── calibration     : .pickle 파일 형식
    ├── depths 
    ├── images 
    ├── npy    
    └── segmentations
    형태로 되어 있을 때, split을 통해 
    train
    ├── images
    │   ├── adidas_80C_img_01.png
    │   ├── ...
    │   └── bubbledoll_80B_img_03.png
    └── segmentations
        ├── adidas_80C_mask_01.png
        ├── ...
        └── bubbledoll_80B_mask_03.png
    val
    .. (위와 동일한 구조를 갖음)
    test
    .. (위와 동일한 구조를 갖음)
    를 만들어낸다.

    그리고 그 후에 segmentation mask 정보를 이용하여 coco format(json)으로 변환한다.
    활용도 측면에서 이 방식이 가장 깔끔함.
    '''
    def __init__(self, args):
        self.args = args
        self.dataset_path = os.path.join(args.data_root, args.dataset)
        self.file_lst = glob.glob(os.path.join(self.dataset_path, "images/*"))
        self.save_path = os.path.join(args.data_root, args.dataset+'_'+args.type)
        random.seed(args.seed)
        print(f"Random seed : {args.seed}")
        random.shuffle(self.file_lst)

    def split(self):
        train_lst, val_lst, test_lst = np.split(np.array(self.file_lst),
                                                        [int(len(self.file_lst) * (1 - (self.args.ratio[1] + self.args.ratio[2]))), 
                                                        int(len(self.file_lst)* (1 - self.args.ratio[2]))])
        file_dict = {
            "train" : list(train_lst),
            "val" : list(val_lst),
            "test" : list(test_lst)
        }
        os.makedirs(os.path.join(self.save_path, 'annotations'), exist_ok=True)

        for key, files in file_dict.items():
            os.makedirs(os.path.join(self.save_path, key, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, key, 'segmentations'), exist_ok=True)
            print(f"Split {key}")
            for file_path in tqdm(files):
                file_name = os.path.basename(file_path)
                mask_path = os.path.join(self.dataset_path, 'segmentations', file_name.replace('img', 'mask'))
                mask_name = os.path.basename(mask_path)
                shutil.copy(file_path, os.path.join(self.save_path, key, 'images', file_name))
                shutil.copy(mask_path, os.path.join(self.save_path, key, 'segmentations', mask_name))
