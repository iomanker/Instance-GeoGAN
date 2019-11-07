import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
IMG_SIZE = (128,128,3)

import pathlib
import csv
import json
ROOT = pathlib.Path("./Dataset/Dataset-CelebA")
DATASET_ROOT = ROOT/"CelebA"
DATA_ROOT = DATASET_ROOT/"Img"/"img_align_celeba"
DATA_PATH = str(DATA_ROOT)
EVAL_PATH = str(DATASET_ROOT/"Eval"/"list_eval_partition.txt")
ANNO_PATH = str(DATASET_ROOT/"Anno"/"list_attr_celeba.txt")
LANDMARK_PATH = str(ROOT/"all_landmark.json")
SEL_ATTRS = ["Goatee"]

def get_raw_image(Ax_filenames_ds,By_filenames_ds,landmark_dict):
    Ax = []
    By = []
    Ax_landmark = []
    By_landmark = []
    for imagename in Ax_filenames_ds:
        filename = str(DATA_ROOT/imagename)
        im = cv2.imread(filename,cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = decode_img(im)
        Ax.append(im)
        landmark_after = process_landmark(landmark_dict[imagename])
        Ax_landmark.append(landmark_after)
    for imagename in By_filenames_ds:
        filename = str(DATA_ROOT/imagename)
        im = cv2.imread(filename,cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = decode_img(im)
        By.append(im)
        landmark_after = process_landmark(landmark_dict[imagename])
        By_landmark.append(landmark_after)
    return (Ax,Ax_landmark),(By,By_landmark)

def decode_img(img):
    # img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.crop_to_bounding_box(img,20,0,178,178)
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
    img = img - 1
    return img

def process_landmark(landmark):
    landmark = np.array(landmark)
    x0 = landmark[:,0]
    y0 = landmark[:,1]
    x1, y1 = x0,y0-20
    x2, y2 = (x1-88)/88,(y1-88)/88
    return tf.stack([x2,y2],1)

def load_csvdata_weneed():
    Ax_filenames_ds = []
    By_filenames_ds = []
    with open(str(DATASET_ROOT/'Ax.csv')) as myfile:
        rows = csv.reader(myfile)
        for row in rows:
            Ax_filenames_ds = row
    with open(str(DATASET_ROOT/'By.csv')) as myfile:
        rows = csv.reader(myfile)
        for row in rows:
            By_filenames_ds = row
    with open(LANDMARK_PATH) as infile:
        landmark_dict = json.load(infile)
    return Ax_filenames_ds,By_filenames_ds,landmark_dict

# celebA 資料處理
# 要Ax, By, landmark_Ax, landmark_By

def write_into_csv(Ax_filenames_ds,By_filenames_ds):
    Ax_filenames_ds = Ax_filenames[:10000]
    By_filenames_ds = By_filenames[:10000]
    with open(str(DATASET_ROOT/'Ax.csv'),'w',newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(Ax_filenames_ds)
    with open(str(DATASET_ROOT/'By.csv'),'w',newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(By_filenames_ds)

def set_celebA_data():
    # 取得Train,Validation圖片檔案名
    with open(EVAL_PATH) as infile:
        lines = infile.readlines()
        lines = [line.strip() for line in lines]
        train_list = [line.split()[0] for line in lines if line.split()[1] == '0']
        # valid_list = [line.split()[0] for line in lines if line.split()[1] == '1']
    with open(LANDMARK_PATH) as infile:
        landmark_dict = json.load(infile)
    # 篩選有Landmark的圖片檔案名
    train_list = [img for img in train_list if img in landmark_dict]
    # valid_list = [img for img in valid_list if img in landmark_dict]
    
    # 載入屬性標記
    with open(ANNO_PATH) as infile:
        lines = infile.readlines()
        # 所有屬性欄位名稱
        all_attrNames = lines[1].split()
        attribute_dict = {}
        # SEL_ATTRS位在該檔案屬性欄位Index
        selected_attribute_index = [all_attrNames.index(sel_attr) \
                                   for idx, sel_attr in enumerate(SEL_ATTRS) \
                                   if sel_attr in all_attrNames]
        selected_attribute_index = np.array(selected_attribute_index)
        # 檔名迭代
        for line in lines[2:]:
            splits = line.split()
            # 取得所有欄位戳記
            attribute_value = [int(x) for x in splits[1:]]
            attribute_value = np.array(attribute_value)
            # attribute_value[attribute_value == -1] = 0
            # 選擇SEL_ATTRS
            attribute_dict[splits[0]] = attribute_value[selected_attribute_index]
    attribute_dict = {img: attribute_dict[img] \
                      for img in attribute_dict \
                      if  img in landmark_dict}
    train_attribute_dict = {img: attribute_dict[img]\
                            for img in attribute_dict
                            if img in train_list}
    # Filter into Ax and By
    Ax_list = []
    By_list = []
    for fname in train_attribute_dict:
        if train_attribute_dict[fname] == 1:
            By_list.append(fname)
        else:
            Ax_list.append(fname)
    return Ax_list,By_list,landmark_dict