import numpy as np
import random
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
model_num = 1
EPOCHS = 2
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
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
print('total picked data :',len(image_list)) #n of all data
#print(len(image_list[0])) #column of one image
#print(len(image_list[0][0])) #row if one image 
#print(image_list[0][0])
image_size = len(image_list) # n of data 

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
print('maybe pkl file order => ',label_binarizer.classes_)

#conbine = list(zip(image_list, y_train))
#random.shuffle(conbine)
#x_train, y_train = zip(*conbine)
#x_train = list(image_list)
#y_train = list(y_train)

x_train = np.array(image_list, dtype=np.float16) / 225.0  #scailing 0~1

# ___________TEST SET ___________________

image_list, label_list = [], []
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
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(plant_folder)
    print("[INFO] Image loading completed")

except Exception as e:
    print(f"Error : {e}")
print('total picked data :',len(image_list)) #n of all data
y_test = label_binarizer.transform(label_list)
x_test = np.array(image_list, dtype=np.float16) / 225.0  #scailing 0~1
#____________________________________

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2,horizontal_flip=True,
    fill_mode="nearest")

print('x_train data shape',x_train.shape)
print('x_test data shape',x_test.shape)
print('y_train data shape',len(y_train))
print('y_test data shape',len(y_test))

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS, epochs=EPOCHS, verbose=1)

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)

print('score(loss,acc) :', scores)
test_pred = model.predict(x_test)
#print(np.argmax(test_pred,axis=1))
#print(np.argmax(y_test,axis=1))

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

model.save('model_p%d.h5'%model_num)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.savefig('acc_p%d.png'%model_num)

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
#plt.show()
plt.savefig('loss_p%d.png'%model_num)

from keras.utils.vis_utils import plot_model
plot_name = 'model_plot_p%d.png'%model_num
plot_model(model, to_file=plot_name, show_shapes=True, show_layer_names=True)


"""
##### use saved model
print('running saved model----------*')
from keras.models import load_model

s_model= load_model('tea_model02.h5')
data = []
data01 = convert_image_to_array('/home/ubuntu/work_/green_tea_model/test_data/test01.JPG')
data02 = convert_image_to_array('/home/ubuntu/work_/green_tea_model/test_data/test02.JPG')
data03 = convert_image_to_array('/home/ubuntu/work_/green_tea_model/test_data/test03.JPG')
data.append(data01)
data.append(data02)
data.append(data03)
data = np.array(data)/225.0
print(data.shape)
test_pred = s_model.predict(data)
print(np.argmax(test_pred,axis=1))
print(test_pred)
print('--------------pixel-------')
print(data)
"""
