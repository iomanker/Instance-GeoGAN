# http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
# * linear resize
# * body in center
# * boarder padding
# * reflection
# * scikit image warp
# "keypoints": [
#     "nose","left_eye","right_eye","left_ear","right_ear",
#     "left_shoulder","right_shoulder","left_elbow","right_elbow",
#     "left_wrist","right_wrist","left_hip","right_hip",
#     "left_knee","right_knee","left_ankle","right_ankle"
# ]
import json
import pathlib
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path("./Dataset")
DATASET_ROOT = ROOT/"coco2017"
DATA_ROOT = DATASET_ROOT/"train2017"
SKEL_ROOT = DATASET_ROOT/"skeleton_train2017"
ANNO_ROOT = DATASET_ROOT/"annotations"

KEYPOINTS_PATH = str(ANNO_ROOT/"person_keypoints_train2017.json")
DATA_PATH = str(DATA_ROOT)
SKELETON_PATH = str(ANNO_ROOT/"skeleton_train2017.json")
SKELETON_PATH_RELATED = str(ANNO_ROOT/"skeleton_train2017_related.json")

def create_skeleton_index():
    images_dict = {}
    output_arr = []
    # handling image ID & skeleton keypoints
    with open(KEYPOINTS_PATH) as json_file:
        logging.info("opened ./{}".format(KEYPOINTS_PATH))
        data = json.load(json_file)

        logging.info("indexing all images' info.")
        for arr in data['images']:
            images_dict[arr['id']] = {itemname: arr[itemname] for itemname in ['file_name','height','width']}

        logging.info("linking categories.")
        for arr in data['annotations']:
            if arr['image_id'] in images_dict and\
               arr['num_keypoints'] == 17:
                for i in range(2,17*3,3):
                    if arr['keypoints'][i] != 2:
                        break
                else:
                    tmp_dict = images_dict[arr['image_id']]
                    tmp_dict['image_id'] = arr['image_id']
                    tmp_dict['keypoints'] = arr['keypoints']
                    tmp_dict['bbox'] = arr['bbox']
                    output_arr.append(tmp_dict)
        
        with open(SKELETON_PATH,'w') as save_file:
            json.dump(output_arr,save_file)
        logging.info("saved ./{}".format(SKELETON_PATH))

def skeleton_related_coordinate(bbox,keypoints):
    output_arr = []
    for i in range(0,17*3,3):
        output_arr.append(keypoints[i]-bbox[0])
        output_arr.append(keypoints[i+1]-bbox[1])
        output_arr.append(keypoints[i+2])
    return output_arr

def process_skeleton_image():
    with open(SKELETON_PATH) as json_file:
        output_arr = []
        data = json.load(json_file)
        logging.info("opened ./{}".format(SKELETON_PATH))
        skeleton_id = 0
        for one_dict in data:
            # keypoints, file_name, bbox, height, width, image_id
            bbox = one_dict['bbox']
            x, y, width, height = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))
            file_name = one_dict['file_name']
            str_skeleton_id = str(skeleton_id).zfill(12)+".jpg"
            image = cv2.imread(str(DATA_ROOT/file_name))
            crop_img = image[y:y+height, x:x+width]
            cv2.imwrite(str(SKEL_ROOT/str_skeleton_id), crop_img)
            logging.info("saved ./{}".format(str(SKEL_ROOT/str_skeleton_id)))

            one_dict['keypoints'] = skeleton_related_coordinate((x, y, width, height),one_dict['keypoints'])
            one_dict['skeleton_id'] = str_skeleton_id
            logging.info("modified keypoints")
            output_arr.append(one_dict)
            skeleton_id += 1

        with open(SKELETON_PATH_RELATED,'w') as save_file:
            json.dump(output_arr,save_file)
            logging.info("saved ./{}".format(SKELETON_PATH_RELATED))

def resize_skeleton_image(img,landmarks,newsize):
    ori_h, ori_w, _ = img.shape
    new_h, new_w, _ = newsize
    img = cv2.resize(img,dsize=(new_w,new_h),interpolation=cv2.INTER_LINEAR)
    scale_h = float(new_h) / float(ori_h)
    scale_w = float(new_w) / float(ori_w)
    return_lm = None
    for idx in range(0,len(landmarks),3):
        if idx == 0:
            return_lm = np.array([landmarks[idx] * scale_w,landmarks[idx+1] * scale_h])
        else:
            return_lm = np.append(return_lm,[landmarks[idx] * scale_w,landmarks[idx+1] * scale_h])
    return img,return_lm

def get_skeleton_imgs_info(filepath):
    # I taught best width, height are 140,280 seperately.
    with open(filepath) as json_file:
        data = json.load(json_file)
        ws, hs = [],[]
        total_w,total_h = 0,0
        total_num = 0
        for one_dict in data:
            bbox = one_dict['bbox']
            x, y, width, height = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))
            if height > 100:
                total_w += width
                total_h += height
                total_num += 1
            ws.append(width)
            hs.append(height)

        avg_w = total_w / total_num
        avg_h = total_h / total_num
        logging.info("avg width:{},height:{}".format(avg_w,avg_h))
        plt.scatter(ws, hs)
        plt.savefig('coco2017_skeleton_scatter.png')

import os.path
import glob
def get_files_and_landmarks_list(filepath,imgroot):
    # img.shape = (height,width,channels)
    filenames = []
    landmarks = []
    with open(str(filepath)) as json_file:
        data = json.load(json_file)
        for one_dict in data:
            filename = str(imgroot/one_dict['skeleton_id'])
            if os.path.exists(filename):
                filenames.append(filename)
            else:
                continue
            keypoints = one_dict['keypoints']
            landmark = []
            for idx in range(0,len(keypoints),3):
                #                width(x)       height(y)
                landmark.append([keypoints[idx],keypoints[idx+1]])
            landmarks.append(landmark)
    return filenames,landmarks
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # create_skeleton_index()
    # process_skeleton_image()
    # resize_skeleton_image()
    # tf.data
    # get_skeleton_imgs_info(SKELETON_PATH_RELATED)