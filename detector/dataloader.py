import torch
from coco_utils import get_coco
from transforms import DetectionPresetEval, DetectionPresetTrain 
import os

def get_dataset(dataset_name, image_set, transform, data_path):
    paths = {dataset_name: (data_path, get_coco)}
    p, ds_fn = paths[dataset_name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds

def get_transform(train):
    if train:
        return DetectionPresetEval()
    else:
        return DetectionPresetEval()

def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(args):
    '''
    학습에 사용할 Dataset, Dataloader를 불러오는 부분.  

    transform의 경우; 일반적인 image transform만 진행하는 것이 아닌 target(mask or keypoint)에 대해서도 진행해야하므로 
    기존 torchvision에 내장되어 있는 함수를 사용하는 것이 아닌 조금 변경하여 사용.

    현재 train, validation 모두 동일한 transform 적용. 다양한 기법 추가 예정.(mask, keypoint 각각에 대해..)
    '''
    dataset_path = os.path.join(args.data_root, '{}_{}'.format(args.dataset, args.type))
    trainset = get_dataset(dataset_name=args.dataset, image_set='train', transform=get_transform(True), data_path=dataset_path)
    validset = get_dataset(dataset_name=args.dataset, image_set="val", transform=get_transform(False), data_path=dataset_path)

    train_sampler = torch.utils.data.RandomSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(validset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=collate_fn
    )
    return train_loader, valid_loader