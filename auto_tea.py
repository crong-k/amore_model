import numpy as np
import random
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelEncoder
from autokeras import ImageClassifier
#from sklearn.preprocessing import LabelBinarizer
#from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation, Flatten, Dropout, Dense
#from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
#from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
#from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt



#BS = 32
default_image_size = tuple((256, 256))
#image_size = 0
directory_root = 'gt_data/gt_train'
directory_root_test = 'gt_data/gt_test'

width=256
height=256
depth=3

# Function to convert images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

# ________TRAIN SET ____________
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)  #[tomato__,pepper__, ....]
    for directory in root_dir :
        if directory == ".DS_Store" :
            root_dir.remove(directory)
    for plant_folder in root_dir :
        plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}")
        print(plant_folder,'=>',len(plant_disease_image_list))
        for single_plant_disease_image in plant_disease_image_list :
            if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

        for image in plant_disease_image_list: #get n imege data from each plant dir 
            image_directory = f"{directory_root}/{plant_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(plant_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")



#print('total picked data :',len(image_list)) #n of all data
#print(image_list[0])
#print(len(image_list[0])) #column of one image
#print(len(image_list[0][0])) #row if one image 
#print(image_list[0][0])
image_size = len(image_list) # n of data 
label_gener = LabelEncoder()
label_gener.fit(label_list)

print('*****label_gener.classes_ *****',label_gener.classes_)

y_train_ = label_gener.transform(label_list)
y_train = [ [i] for i in y_train_ ] 
y_train = np.array(y_train)
x_train = np.array(image_list) / 225.0  #scailing 0~1

# ___________TEST SET ___________________

test_image_list, test_label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root_test)  #[tomato__,pepper__, ....]
    for directory in root_dir :
        if directory == ".DS_Store" :
            root_dir.remove(directory)
    for plant_folder in root_dir :
        plant_disease_image_list = listdir(f"{directory_root_test}/{plant_folder}")
        print(plant_folder,'=>',len(plant_disease_image_list))
        for single_plant_disease_image in plant_disease_image_list :
            if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

        for image in plant_disease_image_list: #get n imege data from each plant dir
            image_directory = f"{directory_root_test}/{plant_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                test_image_list.append(convert_image_to_array(image_directory))
                test_label_list.append(plant_folder)
    print("[INFO] Image loading completed")

except Exception as e:
    print(f"Error : {e}")


y_test_ = label_gener.transform(test_label_list)
y_test = [ [i] for i in y_test_ ]
y_test = np.array(y_test)

x_test = np.array(test_image_list) / 225.0  #scailing 0~1
#______________________MODEL DEF____________________________




clf = ImageClassifier(verbose=True, path='auto-keras/',searcher_args={'trainer_args':{'max_iter_num':30}})
clf.fit(x_train, y_train, time_limit = 2 * 60 * 60) # hour

print(clf.get_best_model_id())
searcher = clf.load_searcher()
print(searcher.history) #view all model result 

# FINAL train with best model  
clf.final_fit(x_train, y_train, x_test, y_test, retrain=False, trainer_args={'max_iter_num': 3})
print('HIT!!')
y = clf.evaluate(x_test, y_test)
print(y)

clf.save_searcher(searcher)
