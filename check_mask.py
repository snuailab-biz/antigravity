import pandas as pd
import csv
import cv2
import os

root = '/home/ljj/data/anti/valid_2'


df = pd.read_csv(os.path.join(root, 'valid.csv'))


# my_data_list.sort(key=lambda x: x[1])
print('adsf')

ddf = df.drop(columns='ID')
img_lst = []
image_lst = df['Image'].values.tolist()
for img in image_lst:
    img_lst.append(img)

print(img)