import torch
from model_load import get_model
from PIL import Image
import numpy as np
import torchvision
from utils import visualize, draw_masks
from transforms import DetectionPresetEval
from detector_args import DetectorArgs


if __name__ == "__main__":
    '''
    main.py를 실행하기 전 각각의 모델 학습이 잘되었는지 확인하는 단계.
    이 부분이 순조롭게 진행된다면 배포(main.py)에서도 정상 동작할 것으로 보임.
    DetectorArgs에서 type을 수정하여 mask, point에 대한 실험을 각각 하기를 권장함.
    '''
    args = DetectorArgs('test')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    pretrained_model = args.mask_model if args.type=='mask' else args.point_model


    model = get_model(model_type=args.type, device=device, pretrained_model=pretrained_model)
    model.to(device)


    with torch.no_grad():
        model.eval()

    transform = DetectionPresetEval()

    import glob
    images = glob.glob('{}/*'.format(args.test_folder))
    for i, image in enumerate(images):
        # image = cv2.imread(image)
        image = Image.open(image)
        # image = F.to
        image, _ = transform(image, None)
        image = [image.to(device)]
        output = model(image)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        bboxes = [list(map(int, bbox)) for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]

        if args.type =='mask':
            print("Segment Vis Number {}".format(i))
            colors = np.random.randint(0, 255, size=(5, 3))
            colors = [tuple(color) for color in colors]
            final_mask = output[0]['masks'] > 0.55
            final_mask = final_mask.squeeze(1)[:2]

            init_image = torch.tensor((image[0].detach().cpu().numpy() * 255).astype(np.uint8))
            seg_result = draw_masks(
                init_image, 
                final_mask,
                colors=colors,
                alpha=0.65
            )
            image = seg_result.permute(1,2,0).numpy()
            visualize(image, bboxes, None)
        elif args.type =='point':
            print("keypoints Vis Number {}".format(i))
            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])
            image = (image[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            visualize(image, bboxes, keypoints)
    if image is None:
        print("--test_folder 'Your Path' ")
