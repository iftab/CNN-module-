#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os, cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import backend as K
import h5py
from keras.models import model_from_json
from sklearn.metrics import classification_report

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD, RMSprop, adam

PATH = os.getcwd()

# set training data path
Train_path = 'E:\CNN_DATA\Latest_data\CNN_train' # enter the path 
train_batch = os.listdir(Train_path)




img_rows = 50 
img_cols = 150
num_channel = 2 #for gryscale its 2,for color image its 3
num_epoch =1  # number of iteration to be perform

img_traindata_list = [] 

for dataset in train_batch:
    img_list = os.listdir(Train_path + '/' + dataset) 
    for img in img_list:
        input_img = cv2.imread(Train_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (50, 150))
        img_traindata_list.append(input_img_resize)

img_data = np.array(img_traindata_list)
img_data = img_data.astype('float32')
img_data /= 255

if num_channel == 2:
    if K.image_dim_ordering() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)

# assign the class lebel
num_class = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[1:2000] = 0
labels[2001:3315] = 1

names = ['zero', 'one']

# convert class lebel into 1 D using one-hot-encoding
Y = np_utils.to_categorical(labels, num_class)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset for training and validation
x_train,x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15,random_state=2)

input_shape = img_data[0].shape
print(img_data.shape)

# creat the CNN model
model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(num_class))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# train the model
model.fit(x_train, y_train,
          batch_size=100,
          epochs=num_epoch,
          verbose=1,
          validation_data=(x_valid, y_valid),
         )

# evaluate the model 
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Save the report 
model_json = model.to_json()
with open("cnn.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

Test_path='E:\CNN_DATA\dataset_txt_ntxt'
test_batch = os.listdir(Test_path)
img_testdata_list = []
for dataset_test in test_batch:
    img_list_test = os.listdir(Test_path + '/' + dataset_test) 
    for img_test in img_list_test:
        input_img_test = cv2.imread(Test_path + '/' + dataset_test + '/' + img_test)
        input_img_test = cv2.cvtColor(input_img_test, cv2.COLOR_BGR2GRAY)
        input_img_resize_test = cv2.resize(input_img_test, (50, 150))
        img_testdata_list.append(input_img_resize_test)
img_data_test = np.array(img_testdata_list)
img_data_test = img_data_test.astype('float32')
img_data_test /= 255

if num_channel == 2:
    if K.image_dim_ordering() == 'th':
        img_data_test = np.expand_dims(img_data_test, axis=1)
        print(img_data_test.shape)
    else:
        img_data_test = np.expand_dims(img_data_test, axis=4)
        print(img_data_test.shape)

else:
    if K.image_dim_ordering() == 'th':
        img_data_test = np.rollaxis(img_data_test, 3, 1)
        print(img_data_test.shape)
num_class_test = 2
num_of_samples_test =img_data_test.shape[0]
labels_test = np.ones(num_of_samples_test, dtype='int64')

labels_test[1:285] = 0
labels_test[286:760] = 1

names_test = ['zero', 'one']
X_test=img_data_test


# convert class lebel into 1 D using one-hot-encoding
Y_test = np_utils.to_categorical(labels_test, num_class_test)
# Shuffle the dataset
x, y = shuffle(img_data_test, Y_test, random_state=2)
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x, y, verbose=0)
#print(loaded_model.metrics_names[1])
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# prediction of test data
y_predict = model.predict_classes(X_test)
print(len(y_predict))
#print(y_predict)
#print(labels_test)
#print(len(labels_test))
path='E:\CNN_DATA\dataset_txt_ntxt\Predicted_Correct'
correct = np.where(y_predict==labels_test)[0]
print("indices from X_test where image are found to be correct:",correct)
print ("Found %d correct labels" % len(correct))
np.savetxt('correct.txt',correct,delimiter=',')
l=len(correct)
#for  i,correct in enumerate(correct):
   # plt.subplot(3,3,i+1)
   # plt.imshow(X_test[correct].reshape(50,150), cmap='gray', interpolation='none')
   # plt.title("\n Predicted {}, Class {}".format(y_predict[correct], labels_test[correct]))
   # plt.tight_layout()
for m in  range(0,len(correct)):
    restore_img_correct=X_test[correct[m]].reshape(50,150)
    img = cv2.imread('restore_img_correct', 1)
    cv2.imwrite(os.path.join(path , 'm','.png'),img)
    cv2.waitKey(0)

incorrect = np.where(y_predict!=labels_test)[0]
#print ("Found %d incorrect labels"% len(incorrect))
print(" indices of X_test where image are found to be incorrect:",incorrect)
#np.savetxt('incorrect.txt',incorrect.delimiter=',')
#for i, incorrect in enumerate(incorrect[:9]):
  #  plt.subplot(3,3,i+1)
   # plt.imshow(X_test[incorrect].reshape(50,150), cmap='gray', interpolation='none')
   # plt.title("Predicted {}, Class {}".format(y_predict[incorrect], labels_test[incorrect]))
   # plt.tight_layout()
    

target_names = ["Class {}".format(i) for i in range(num_class)]
print(classification_report(labels_test, y_predict, target_names=target_names))









