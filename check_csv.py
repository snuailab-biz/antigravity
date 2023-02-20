import csv
import cv2
import os

root = '/home/ljj/data/anti/valid_2'

with open(os.path.join(root, 'valid.csv'), 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        for k, v in row.items():
            if 'png' in v:
                if 'img' in v:
                    print(v)
                path = os.path.join(root, v)
                cv2.imshow(k, cv2.imread(path))
            if 'Volume' in k:
                print(f'{k} : {v}')
        print("==============================")
        cv2.waitKey(0)