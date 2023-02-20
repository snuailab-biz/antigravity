
from detector.detector_main import AntiDetector
import cv2
if __name__ == '__main__':
    import glob

    # image_path = r'/home/ljj/data/anti/validation_anti/2/images/*'
    image_path = r'/home/ljj/data/anti/testss/*'
    # image_path = r'/home/ljj/antidata/train/images/*'
    images_path = glob.glob(image_path)
    images = []

    detector = AntiDetector(True)
    for i, img_path in enumerate(images_path):
        img = cv2.imread(img_path)
        mask_img, keypoint_img, pred_info = detector.predict(image=img, image_path='')
        # cv2.imshow('mask_img', mask_img)
        # cv2.imshow('keypoint_img', keypoint_img)
        # cv2.waitKey(0)
        

    