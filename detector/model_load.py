import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from .model_u2 import U2NET # full size version 173.6 MB

def get_model(model_type, device, pretrained_model=None):
    '''
    각 mask, point 모델에 대해 load하는 부분.
    기본적으로 torchvision 모델을 가져와 사용한다. 모델을 수정할 수 있으며, torchvision doc 확인.
    torch==1.10+cu113, torchvision==0.11+cu113 에서 동작.
    '''
    if model_type == 'point':
        # 새로운 모델
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0))
        # 기존 제공하였던 모델
        # anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=None,
                                                                    pretrained_backbone=True,
                                                                    num_keypoints=4,
                                                                    num_classes=2, # Background is the first class, object is the second class
                                                                    rpn_anchor_generator=anchor_generator)

    elif model_type == 'mask':
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                           aspect_ratios=(0.05, 0.25, 1.0))
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.05, 0.25, 1.0))
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(rpn_anchor_generator=anchor_generator, pretrained=True)
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        hidden_layer = 256

        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    elif model_type == 'u2':
        model = U2NET(3,1)


    if pretrained_model is not None:
        ckpt = torch.load(pretrained_model, map_location=device)
        model.load_state_dict(ckpt)

    model.to(device)

    return model