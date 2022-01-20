import os, shutil
from numpy import loadtxt



#To use this script, first download Places365 validation dataset

img_table = loadtxt("./data/places365_val.txt", dtype='str')
img_dir = os.listdir("./data/val_256")

class_list = [234, 243, 254, 266, 271, 279, 288, 306, 309, 323, 339, 338, 341, 355, 356, 357, 359, 362, 66, 48, 104, 81,
              110, 74, 62, 138, 116, 117, 142, 190, 94, 97, 78, 140, 141, 142, 150, 151, 152, 173, 184, 197, 204, 205,
              209, 224, 232, 233]

for idx, images in enumerate(img_dir):

    if int(img_table[idx][1]) in class_list:
        shutil.copy(os.path.join("./data/val_256", img_table[idx][0]), './data/val_256_new')

print('done')