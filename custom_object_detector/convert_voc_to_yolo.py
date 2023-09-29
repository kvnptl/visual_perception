import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm

# class_mapping = {'Chihuahua': 0,
#                   'Golden_Retriever': 1,
#                   'Welsh_Springer_Spaniel': 2,
#                   'German_Shepherd': 3,
#                   'Doberman': 4,
#                   'Boxer': 5,
#                   'Siberian_Husky': 6,
#                   'Pug': 7,
#                   'Pomeranian': 8,
#                   'Cardigan': 9}


dirs = ['/home/kpatel2s/kpatel2s/object_detection/custom_object_detector/dataset/standford_dogs_mini_10']
classes = ['Chihuahua', 
           'Golden_retriever', 
           'Welsh_springer_spaniel', 
           'German_shepherd', 
           'Doberman', 
           'Boxer', 
           'Siberian_husky', 
           'Pug', 
           'Pomeranian', 
           'Cardigan']

classes = [c.lower() for c in classes]

remove_files = []

def getImagesInDir(dir_path):
    image_list = []
    img_dir_path = dir_path + '/images'
    # go through all subdirectories
    for subdir in listdir(img_dir_path):
        if not subdir.startswith('._'):
            for filename in listdir(join(img_dir_path, subdir)):
                image_list.append(join(img_dir_path, subdir, filename))
    return image_list

def removeImagesInDir(dir_path):
    image_list = []
    cnt_2 = 0
    img_dir_path = dir_path + '/images'
    annot_dir_path = dir_path + '/annotations'
    # go through all subdirectories
    for subdir in listdir(img_dir_path):
        if not subdir.startswith('._'):
            for filename in listdir(join(img_dir_path, subdir)):
                if filename.split('.')[0] in remove_files:
                    os.remove(join(img_dir_path, subdir, filename))
                    # remove from annotations
                    os.remove(join(annot_dir_path, subdir, filename.replace('.jpg', '')))
                    cnt_2 += 1

    print('removed {} images'.format(cnt_2))

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    # custom code
    subdir_start_name = basename_no_ext.split('_')[0]
    # find the folder name with subdir_start_name
    parent_dir = dir_path + '/' + 'annotations'
    # dir name start with subdir_path

    for dirname in os.listdir(parent_dir):
        if dirname.startswith(subdir_start_name):
            full_subdir = parent_dir + '/' + dirname
            break

    # in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    in_file = open(full_subdir + '/' + basename_no_ext)
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    cnt = 0
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls.lower() not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls.lower())
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        cnt += 1
        if cnt > 1:
            remove_files.append(basename_no_ext)

    in_file.close()
    out_file.close()

# cwd = getcwd()

# for dir_path in dirs:
full_dir_path = dirs[0]
output_path = full_dir_path +'/yolo/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

image_paths = getImagesInDir(full_dir_path)
list_file = open(full_dir_path + '/image_paths'  + '.txt', 'w')

for image_path in tqdm(image_paths):
    list_file.write(image_path + '\n')
    convert_annotation(full_dir_path, output_path, image_path)
list_file.close()

# remove files 
# removeImagesInDir(full_dir_path)

print('Done')

