import glob

def write_train_list(rootDir, revised=True):
    imageDir = rootDir + '/images'
    depthDir = rootDir + '/depths'
    maskDir  = rootDir + '/segmentations'

    imageData = glob.glob(imageDir + '/*.png') or glob.glob(imageDir + '/*.jpg')
    dapthData = glob.glob(depthDir + '/*.png') or glob.glob(depthDir + '/*.jpg')
    maskData  = glob.glob(maskDir + '/*.png') or glob.glob(maskDir + '/*.jpg')

    if revised == True:
        with open('{}/train_file_list.txt'.format(rootDir), 'w') as f:
            for img, dep, seg in zip(imageData, dapthData, maskData):
                image = img.split("\\")[-1]
                depth = dep.split("\\")[-1]
                mask = seg.split("\\")[-1]
                d_focal = 252.1192  # c_focal=607.6263

                f.writelines('{}/{} {}/{} {}/{} {}\n'.format(imageDir, image, depthDir, depth, maskDir, mask, d_focal))
    else:
        with open('{}/train_file_list.txt'.format(rootDir), 'w') as f:
            for img, dep in zip(imageData, dapthData):
                image = img.split("\\")[-1]
                depth = dep.split("\\")[-1]
                d_focal = 252.1192  # c_focal=607.6263
                f.writelines('{}/{} {}/{} {}\n'.format(imageDir, image, depthDir, depth, d_focal))


def write_test_list(rootDir, revised=True):

    imageDir = rootDir + '/images'
    maskDir  = rootDir + '/segmentations'

    imageData = glob.glob(imageDir + '/*.png') or glob.glob(imageDir + '/*.jpg')
    maskData  = glob.glob(maskDir + '/*.png') or glob.glob(maskDir + '/*.jpg')

    if revised == True:
        with open('{}/test_file_list_revised.txt'.format(rootDir), 'w') as f:
            for img, seg in zip(imageData, maskData):
                image = img.split("\\")[-1]
                mask = seg.split("\\")[-1]
                d_focal = 252.1192  # c_focal=607.6263

                f.writelines('{}/{} None {}/{} {}\n'.format(imageDir, image, maskDir, mask, d_focal))
    else:
        with open('{}/test_file_list.txt'.format(rootDir), 'w') as f:
            for img in imageData:
                image = img.split("\\")[-1]
                d_focal = 252.1192  # c_focal=607.6263

                f.writelines('{}/{} None {}\n'.format(imageDir, image, d_focal))


if __name__ == '__main__':

   # train_path = './dataset/dataset_bts_train'
   # write_train_list(train_path, revised=False)

   test_path  = './dataset/dataset_test_iphone'
   write_test_list (test_path, revised=False)
