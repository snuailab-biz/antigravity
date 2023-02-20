import cv2
import torchvision
import numpy as np
import torch
import os
from common import timefn2
from PIL import Image
from .utils import draw_keypoint, draw_mask
from .transforms import DetectionPresetEval, normPRED, RescaleT, ToTensorLab
from .detector_args import DetectorArgs
from .model_load import get_model
from torchvision import transforms#, utils
from torch.autograd import Variable

# 하드코딩
mask_table = (
    # [0, 0, 0],        # bg:
    [33, 33, 33],  # 1: pad l
    [15, 15, 15],  # 2: pad r
    [11, 11, 11],  # 3:
    [28, 28, 28],  # 4:
    [26, 26, 26],  # 5: t1
    [29, 29, 29],  # 6: left1
    [5, 5, 5],  # 7: b1
    [14, 14, 14],  # 11: center
    [1, 1, 1],  # 10: t2
    [13, 13, 13],  # 9:  right
    [6, 6, 6] # 8: b2
)

class AntiDetector(object):
    def __init__(self, DEBUG=False, save_path=None):
        self.DEBUG=DEBUG
        self.args = DetectorArgs(mode='test')
        self.device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

        self.model_mask = get_model(model_type='mask', device=self.device, pretrained_model=self.args.mask_model)
        self.model_u2 = get_model(model_type='u2', device=self.device, pretrained_model=self.args.u2n_model)
        self.model_point = get_model(model_type='point', device=self.device, pretrained_model=self.args.point_model)

        self.transform = DetectionPresetEval() # rcnn계열 모델 학습에 사용된 transform 
        self.transform_u2=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]) # u2 모델 학습에 사용된 transform

        self.save_path = save_path # visualization이 아닌 저장을 위한 path


    @timefn2
    @torch.no_grad()
    def predict(self, image, filename=None):
        self.model_point.eval()
        self.model_mask.eval()
        self.model_u2.eval()

        self.mask_img = np.zeros(image.shape, dtype=np.uint8)
        self.keypoint_img = np.zeros(image.shape, dtype=np.uint8)

        # step1
        self.predict_maskrcnn(image)

        # step2
        self.predict_u2net(image)

        # step3
        self.predict_point(image)

        if self.save_path:
            cv2.imwrite(os.path.join(self.save_path, 'img', filename), image)
            cv2.imwrite(os.path.join(self.save_path, 'mask', filename.replace('img', 'mask')), self.mask_vis)
            cv2.imwrite(os.path.join(self.save_path, 'point', filename.replace('img', 'point')), self.point_vis)

        if self.DEBUG:
            cv2.imshow('Mask Image_u2', self.mask_vis)
            cv2.imshow('Landmark Image', self.point_vis)
            cv2.waitKey(0)
        
        pred_info = {'bbox': self.boxes, 'landmark': self.keypoints}
        return self.mask_img, self.keypoint_img, pred_info


    # Step 1
    @timefn2
    def predict_maskrcnn(self, image):
        '''
        볼륨을 측정하기 위한 마스크를 찾거나 박스를 찾는 것이 아닌 u2net에서 필요한 것을 얻기 위해 진행.
            ;self.roi
            ;self.center_point 
        '''
        image_inp, _ = self.transform(image, None)
        image_inp = [image_inp.to(self.device)]
        output = self.model_mask(image_inp)[0]

        scores = output['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
        bboxes = [list(map(int, bbox)) for bbox in output['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()][:2]

        bbox_arr = np.array(bboxes)
        offset = 15 # Pad영역을 잘못 찾게 되어 실제 영역보다 작게 뽑게 될 경우 이후 u2net 과정에서 마스크가 잘리는 경우가 발생하여 offset을 넣음.
        box_y = bbox_arr[:, 1::2]
        box_x = bbox_arr[:, 0::2]
        roi_arr = np.array([np.min(box_x)-offset, np.min(box_y)-offset, np.max(box_x)+offset, np.max(box_y)+offset])
        self.roi = np.where(roi_arr < 0, 0, roi_arr) # maskrcnn은 pad영역을 확실하게 찾으므로 이 영역을 통해 u2net에서 패드 위치를 정하는데 필요.
        
        final_mask = output['masks'] > 0.5 # threshold 값으로 학습이 잘된 경우에는 높여 사용하여도 됨. 상위 2개의 마스크만 뽑기 때문에 큰 의미는 없음.
        final_mask = final_mask.squeeze(1)[:2]

        mask1 = final_mask[0].detach().cpu().numpy()
        mask2 = final_mask[1].detach().cpu().numpy()
        mask1_min = np.min(np.argwhere(mask1>0)[:,1])
        mask2_min = np.min(np.argwhere(mask2>0)[:,1])
        if mask1_min < mask2_min:
            left_box = bboxes[0]
            right_box = bboxes[1]
        else:
            right_box = bboxes[0]
            left_box = bboxes[1]
        self.center_point = (left_box[2]+right_box[0])//2


    # Step 2
    @timefn2
    def predict_u2net(self, image_BGR, lower=(170,170,170)): # lower는 cv2.inRange 색상 범위. 
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB) # u2net을 학습 시키는 경우는 io.imread로 로드, opencv load는 BGR. RGB 변환.

        # u2 transform 입력형태 
        shape=image.shape
        label=np.zeros(shape)
        idx = np.array([1])
        sample = {'imidx':np.array([idx]), 'image':image, 'label':label}
        ######################

        # u2 Result
        inp = self.transform_u2(sample)
        inp = inp['image'].unsqueeze(0).type(torch.FloatTensor)
        inp = Variable(inp.cuda())
        d1,d2,d3,d4,d5,d6,d7= self.model_u2(inp)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy() # Segment 
        im = Image.fromarray(predict_np*255).convert('RGB')
        im = im.resize((shape[1], shape[0]), resample=Image.BILINEAR)
        im_array = np.array(im)
        #####

        # Probabiltiy 형태의 이미지가 나오므로 패드부분을 찾을 수 있게 제거 
        res_mask = cv2.inRange(im_array, lower, (255,255,255))
        roi_mask = np.zeros(res_mask.shape, dtype=np.uint8)
        roi_mask[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] = res_mask[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] # RoI 는 maskrcnn에서 찾은 pad 영역

        contr1, contr2, bbox1, bbox2 = self.get_contours(roi_mask) # point를 마스크의 끝라인에 맞추기 위해 컨투어를 진행하여 끝 선 얻기.
        
        # 결과로 나온 영역에서 왼쪽, 오른쪽이 구분되어지지 않기 때문에 2d영역상에서 왼쪽에 있는 패드를 왼쪽 패드, 오른쪽에 있는 패드를 오른쪽 패드로 만들기.
        if bbox1[0] < bbox2[0]:
            left_box = [bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]]
            right_box = [bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]]
            self.left_contr = contr1
            self.right_contr = contr2
        else:
            left_box = [bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]]
            right_box = [bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]]
            self.left_contr = contr2
            self.right_contr = contr1

        left_mask = np.zeros(res_mask.shape)
        right_mask = np.zeros(res_mask.shape)
        left_mask[left_box[1]:left_box[3],left_box[0]:left_box[2]] = res_mask[left_box[1]:left_box[3],left_box[0]:left_box[2]]
        right_mask[right_box[1]:right_box[3],right_box[0]:right_box[2]] = res_mask[right_box[1]:right_box[3],right_box[0]:right_box[2]]

        # Visualization을 위한 패드 영역 그리기
        self.mask_vis = draw_mask(image.copy(), (left_mask, right_mask), 0.3, colors=[(255,0,0), (0,255,0)])

        # volume측정에서 사용될 패드 영역 채우기
        self.mask_img[left_mask>0,:] = np.array(mask_table[0],dtype=np.uint8)[None,:]
        self.mask_img[right_mask>0,:] = np.array(mask_table[1],dtype=np.uint8)[None,:]

        # box정보는 이후 사용되지 않지만 information로 가지고 있음.
        self.boxes = {'left': left_box, 'right':right_box}
    
    @timefn2
    def predict_point(self, image):
        image, _ = self.transform(image, None)
        image = [image.to(self.device)]

        output = self.model_point(image)[0]
        scores = output['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.6)[0].tolist() # Indexes of boxes with scores > 0.6
        post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
        kps = output['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()[:2]
        keypoint1 = kps[0][:,:2]
        keypoint2 = kps[1][:,:2]
        #

        # 뽑은 keypoint(landmark)를 top, left, bottom, right로 정렬.
        t1, l1, b1, r1 = self.get_keypoint_locate(keypoint1)
        t2, l2, b2, r2 = self.get_keypoint_locate(keypoint2)
        # top기준으로 왼쪽에 있으면 왼쪽 패드, 오른쪽이면 오른쪽 패드
        if t1[0] < t2[0]:
            left_kp = np.array([t1,l1,b1,r1])
            right_kp = np.array([t2,l2,b2,r2])
        else:
            left_kp = np.array([t2,l2,b2,r2])
            right_kp = np.array([t1,l1,b1,r1])
        
        # u2net에서 찾은 컨투어 기준으로 landmark 위치 정렬.
        left_kp = self.get_neighbor_point(left_kp, self.left_contr[:,0,:])
        right_kp = self.get_neighbor_point(right_kp, self.right_contr[:,0,:])

        self.keypoints = {'left': left_kp, 'right': right_kp}

        # Visualization을 위한 그리기
        image = (image[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        self.point_vis = draw_keypoint(image, None, self.keypoints)

        # 볼륨 측정에 사용될 수 있는 landmakr image, 현재는 information(좌표정보)을 사용.
        for locate, kps in self.keypoints.items():
            for idx, kp in enumerate(kps):
                cv2.circle(self.keypoint_img, tuple([int(kp[0]), int(kp[1])]), 5, mask_table[idx+4], -1)
    
    def get_contours(self, mask):
        '''
        컨투어가 두 개 이상일 경우;
            전체 영역에서 패드가 많은 부분을 차지 하기 때문에 크기 정렬을 통해 큰 컨투어 두개만 추출
        컨투어가 두 개 이하일 경우;
            두 개 이하인 경우는 가운데 부분을 분할하지 못하고 브라 전체를 찾게 되어 발생하여 왼쪽 오른쪽을 구분하지 못하는 경우가 발생함.
            maskrcnn을 통해 왼쪽 패드와 오른쪽 패드의 끝점을 얻고 그 점의 중앙이 나누어지는 부분이라 생각하여 중앙점 기준 y축을 0으로 만들고 다시 컨투어를 진행.
            이러한 경우 심한 오차가 발생할 수 있음.
        '''
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 2:
            length = [len(contr) for contr in contours]
            sort_lst = sorted(range(len(length)), key=lambda k: length[k], reverse=True)
            contours = (contours[sort_lst[0]], contours[sort_lst[1]])

        elif len(contours) < 2: 
            mask[:,self.center_point] = 0
            return self.get_contours(mask)

        contr1 = contours[0]
        contr2 = contours[1]
        bbox1 = cv2.boundingRect(contr1)
        bbox2 = cv2.boundingRect(contr2)

        return contr1, contr2, bbox1, bbox2


    def get_neighbor_point(self, points, contr):
        '''
        Landmark detection을 통해 찾게 된 포인트들을 컨투어 라인의 가까운 점에 위치하게 조정.
            landmark가 패드영역 바깥쪽일 경우 패드영역에 붙게하고, 
            landmark가 패드영역 안쪽인 경우 패드 영역 끝에 붙게 만듬.
        '''
        near_keypoints = []
        for kp in points:
            kp = np.array([int(kp[0]), int(kp[1])])
            distance = np.linalg.norm(contr-kp, axis=1)
            min_arg = np.argmin(distance)
            near_keypoints.append(tuple(contr[min_arg][:2].tolist()))
        return np.array(near_keypoints)




    def get_keypoint_locate(self, keypoint):
        '''
        landmark detector가 찾은 point들의 순서를 top, left, bottom, right로 변환
        패드 자체의 top, left, bottom, right로 정하는 게 아닌 보이는 관점(image상)에서의 위치
            수직으로 촬영할 경우 잘못 나올 수 있음.
            왼쪽 아래 오른쪽의 포인트가 바닥에 깔리는 모양의 브라는 이 방법이 문제가 될 수 있음.
        '''
        kp = keypoint.copy()
        left_ind = np.argmin(kp[:,0])
        left = kp[left_ind]
        kp = np.delete(kp, left_ind, axis=0)
        right_ind = np.argmax(kp[:,0])
        right = kp[right_ind]
        kp = np.delete(kp, right_ind, axis=0)
        top_ind = np.argmin(kp[:,1])
        top = kp[top_ind]
        kp = np.delete(kp, top_ind, axis=0)
        bottom_ind = np.argmax(kp[:,1])
        bottom = kp[bottom_ind]
        kp = np.delete(kp, bottom_ind, axis=0)
        return top, left, bottom, right






