### IMPORTS
import os
import numpy as np   
from PIL import Image
from random import randint
import cv2
from shutil import copyfile


### FUNCTIONS
def flip_image(image_path, saved_location):
    ''' horizontal mirroring'''
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)
    return image_obj.size

def normalize_labels(path):
    '''fix label value range between 0 and 1'''
    for file in os.listdir(path):
        if file.endswith(".txt"):
            continue
        pic_path = path + file
        pic_desc = path + file.split('.')[0] + '.txt'
        with open(pic_desc) as desc_file:
            category,x1,y1,x2,y2 =[float(x) for x in desc_file.readline().split(' ')]
        image_obj = Image.open(pic_path)
        width, height = image_obj.size
        x1 = x1 / width
        x2 = x2 / width
        y1 = y1 / height
        y2 = y2 / height
        with open(pic_desc, "w+") as desc_file:
            desc_file.write(' '.join([str(category),str(x1),
                    str(y1),str(x2),str(y2)])) 

def preprocess_mirrored(path,savePath):
    '''generate mirrored images and labels'''
    for file in os.listdir(path):
        if file.endswith(".txt"):
            continue
        # prepare variables
        pic_path = path + file
        pic_desc = path + file.split('.')[0] + '.txt'
        with open(pic_desc) as desc_file:
            category,x1,y1,x2,y2 =[int(x) for x in desc_file.readline().split(' ')]
        tmp = file.split('.')
        new_file_name = savePath + tmp[0] + '_mirrored.' + tmp[1]
        new_file_label = savePath + tmp[0] + '_mirrored.txt'
        # flip and save image
        width, height = flip_image(pic_path, new_file_name)
        # flip and save new labels
        with open(new_file_label,"w+") as desc_file:
            desc_file.write(' '.join([str(category),str(width - x2),
                    str(y1),str(width - x1),str(y2)])) 

def make_validation_set(path, out_path, count):
    '''move "count" random elements from path to out_path
    including the .txt '''
    files = os.listdir(path)
    sample_count = len(files)/2
    for iter in range(count):
        val = randint(0,sample_count) * 2
        os.rename(path + files[val], out_path + files[val])
        os.rename(path + files[val+1], out_path + files[val+1])
        sample_count -= 1
        files.pop(val)
        files.pop(val)
    
def rescale(path, out_path, size):
    '''downscale images
    size - tuple (width, height)'''
    files = os.listdir(path)
    for i in range(len(files)):
        if files[i].endswith('.jpg'):
            new_img = cv2.cvtColor(cv2.resize(cv2.imread(path + files[i], cv2.IMREAD_COLOR), size,
             interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_path + files[i], new_img)
            copyfile(path + files[i+1],out_path + files[i+1])

def folders_for_imagegen(path):
    ''' move images to folders by their categories'''
    files = os.listdir(path)
    if not os.path.exists(path + '0'):
        os.mkdir(path + '0')
    if not os.path.exists(path + '1'):
        os.mkdir(path + '1')
    for i in range(len(files)):
        if files[i].endswith('.txt'):
            val = np.loadtxt(path + files[i])
            if val[0] == 1:
                os.rename(path + files[i-1], path + '0\\' + files[i-1])
            else:
                os.rename(path + files[i-1], path + '1\\' + files[i-1])

### MAIN
if __name__ == "__main__":
    #rescale(validation_set, 'data\\valid_0.9_150x150\\')
    #make_validation_set('data\\123\\','data\\valid\\',600)
    #folders_for_imagegen('data\\valid\\')
    #preprocess_mirrored('data\\original\\')
    #make_validation_set('data\\mirrored\\','data\\valid\\',600)
    normalize_labels('data\\valid\\')