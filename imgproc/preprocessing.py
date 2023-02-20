import glob
import os

import cv2
import numpy as np

from common.common import timefn2


class PreProcessing(object):

    def __init__(self):
        self.preporc = 'preporcessing'

    @timefn2
    def depth_to_gray_scale(self, indir, outdir):

        img_list = indir + '/*.png'
        depth_img_list = glob.glob(img_list)

        for depth in depth_img_list:
            img = cv2.imread(depth)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if not os.path.isdir(outdir):
                os.mkdir(outdir)

            filepath = '{}/{}'.format(outdir, depth.split("\\")[-1])
            cv2.imwrite(filepath, gray)

    @timefn2
    def gray_mask_to_color_mask(self, indir, outdir, dtype):
        seg_list = glob.glob('{}/*.png'.format(indir))

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        mask_table, color_table = self.get_pad_color_table()
        for filename in seg_list:
            seg = cv2.imread(filename)
            img = np.zeros(shape=seg.shape, dtype=np.uint8)

            if dtype == 'DPT':
                color = [150, 5, 61]  # RGB
                for mask in mask_table:
                    if not (mask == [14, 14, 14]).all():  # 11번 landmark 제외
                        img[np.where((seg == mask).all(axis=2))] = color
            elif dtype == 'NoMask':
                pass
            else:
                for mask, color in zip(mask_table, color_table):
                    img[np.where((seg == mask).all(axis=2))] = color

            img_filename = '{}/{}'.format(outdir, filename.split('\\')[-1])
            img = img.astype(np.uint8)
            cv2.imwrite(img_filename, img[:, :, ::-1])

    @timefn2
    def get_pad_color_table(self):

        mask_table = np.asarray([
            # [0, 0, 0],        # bg:
            [33, 33, 33],  # 1:
            [15, 15, 15],  # 2:
            [11, 11, 11],  # 3:
            [28, 28, 28],  # 4:
            [26, 26, 26],  # 5:
            [29, 29, 29],  # 6:
            [5, 5, 5],  # 7:
            [6, 6, 6],  # 8:
            [13, 13, 13],  # 9:
            [1, 1, 1],  # 10:
            [14, 14, 14]  # 11:
        ])

        color_table = np.asarray([
            # [0, 0, 0],          # bg:
            [119, 11, 32],  # 1:
            [150, 100, 100],  # 2
            [70, 70, 70],  # 3
            [0, 60, 100],  # 4
            [0, 0, 142],  # 5
            [0, 0, 90],  # 6
            [111, 74, 0],  # 7
            [255, 215, 0],  # 8
            [190, 153, 153],  # 9
            [81, 0, 81],  # 10
            [180, 165, 180],  # 11
        ])

        return mask_table, color_table

    def _get_pad_table_img(self):
        _, colors = self.get_pad_color_table()
        width = 100
        height = 100
        tableimg = []
        for i in range(len(colors)):
            img = np.ones([width, height, 3]) * colors[i]
            tableimg.append(img)

        tableimg = np.concatenate(tableimg, axis=1)
        import cv2
        tableimg = tableimg.astype(np.uint8)
        cv2.imwrite("colortable.png", tableimg[:, :, ::-1])

    @timefn2
    def image_resize_preserve_ratio(self, image_path, out_path=None, size=(512, 288), DEBUG=False):

        files = glob.glob('{}/*.png'.format(image_path)) or glob.glob('{}/*.jpg'.format(image_path))
        images = []
        for img in files:
            base_pic = np.ones((size[1], size[0], 3), np.uint8) * 255
            pic1 = cv2.imread(img, cv2.IMREAD_COLOR)
            h, w = pic1.shape[:2]
            ash = size[1] / h
            asw = size[0] / w
            if asw < ash:
                sizeas = (int(w * asw), int(h * asw))
            else:
                sizeas = (int(w * ash), int(h * ash))
            pic1 = cv2.resize(pic1, dsize=sizeas)
            base_pic[int(size[1] / 2 - sizeas[1] / 2):int(size[1] / 2 + sizeas[1] / 2),
            int(size[0] / 2 - sizeas[0] / 2):int(size[0] / 2 + sizeas[0] / 2), :] = pic1

            if DEBUG:
                if not os.path.isdir(out_path):
                    os.mkdir(out_path)
                cv2.imwrite('{}/{}'.format(out_path, img.split('\\')[-1]), base_pic)
            images.append(base_pic)

        return images, files

    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)
