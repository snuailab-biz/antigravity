from utils_dataset import split_dataset
from utils_segm import mask2coco
from detector_args import DetectorArgs

if __name__ == "__main__":
    '''
    Split & Format Convert
    각 Class에 대한 주석 확인.
    '''
    # split Dataset (train, validation, test)
    args = DetectorArgs(mode='dataset')
    # print(args.type)
    # split_dataset = split_dataset(args)
    # split_dataset.split()

    # Data format (img, mask) to coco format(json)

    mask2coco_train = mask2coco(args, 'train')
    mask2coco_train.convert() # # mask2coco_train.visualize()

    # mask2coco_train.visualize()

    mask2coco_val = mask2coco(args, 'val')
    mask2coco_val.convert()
    # mask2coco_val.visualize()

    mask2coco_test = mask2coco(args, 'test')
    mask2coco_test.convert()
    mask2coco_test.visualize() # 데이터 확인 



