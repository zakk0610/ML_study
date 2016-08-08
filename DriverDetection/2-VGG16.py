import sys  
import os
from os import listdir
from os.path import isfile, join
import cv2
import glob
import math
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from PIL import Image

data_path = os.getcwd()

batch_size = 46
nb_classes = 10
nb_epoch = 74

# input image dimensions
img_rows, img_cols = 224, 224

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_cols,img_rows)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(weights_path)

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return model


def cache_data(data1, data2, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    np.savez(path, d1=data1, d2=data2)

def restore_data(path):
    if os.path.isfile(path):
        print('Restore data from cache........')
        data = np.load(path)
        return data['d1'], data['d2']
    else:
        print('restore data failed')

def read_and_normalize_training_data(size, data_set, img_rows, img_cols):
    data_set_path = data_path + '/' + data_set
    test_data = np.zeros((size, 3, img_cols, img_rows),  dtype="uint8")
    test_id = np.zeros((size, 1), dtype="uint8") 
    print ('prepare ' + data_set)
    cache_path = os.path.join('cache', 'zakk' + data_set +str(img_rows)+'_'+str(img_cols)+'.npz')
    if not os.path.isfile(cache_path):
        idx = 0
        for i in range(10):
            img_path = data_set_path+'/'+str(i)
            files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
            for j in range (len(files)):
                img = Image.open(img_path+'/'+files[j])
                imgarr = np.asarray(img)
                test_data[idx,0,:,:] = imgarr[:,:,0]
                test_data[idx,1,:,:] = imgarr[:,:,1]
                test_data[idx,2,:,:] = imgarr[:,:,2]
                test_id[idx:] = i
                idx+=1
        test_data.astype('float16')
        # convert class vectors to binary class matrices
        test_id = np_utils.to_categorical(test_id, nb_classes)
        mean_pixel = [103.939, 116.779, 123.68]
        for c in range(3):
            test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
        cache_data(test_data, test_id, cache_path)
    else:
        print('restore train from cache!')
        (test_data, test_id)  = restore_data(cache_path)
    return test_data, test_id

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def load_test(testing_set, img_rows, img_cols, color_type=3):
    print('Read test images')
    path = os.path.join('.', 'picture', testing_set, '*.jpg')
    print(path)
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    return X_test, X_test_id

def old_cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def old_restore_data(path):
    data = dict()
    print('Restore data from pickle........')
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def read_and_normalize_test_data(testing_set, img_rows=224, img_cols=224, color_type=3):
    cache_path = os.path.join('cache', 'zakktesting_set_' + testing_set +str(img_rows) +
                              '_' + str(img_cols) + '.dat')
    if not os.path.isfile(cache_path):
        test_data, test_id = load_test(testing_set, img_rows, img_cols, color_type)
        old_cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = old_restore_data(cache_path)
    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def test_model_and_submit(testing_set, model, start=1, end=1, modelStr=''):
    img_rows, img_cols = 224, 224

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(testing_set, img_rows, img_cols)
                                                      
    yfull_test = []

    for index in range(start, end + 1):
        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        yfull_test.append(test_prediction)

    info_string = 'zakk_' + testing_set +  modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)

# read training data (0-21600)
#X_train, Y_train = read_and_normalize_training_data(21601, 'training_set', img_rows, img_cols)
# read validation data (21601-22423)
#X_test, Y_test = read_and_normalize_training_data(823, 'validation_set', img_rows, img_cols)

#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

#model = VGG_16()
model = VGG_16("web_vgg16_weights.h5")

model.summary()

#early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(X_test, Y_test), shuffle=True, callbacks=[early_stopping])

#model.save_weights('VGG16_weights.h5')
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])


model.load_weights('VGG16_weights.h5')
test_model_and_submit('testing1', model, 1, 2, 'myVGG16')
test_model_and_submit('testing2', model, 1, 2, 'myVGG16')
test_model_and_submit('testing3', model, 1, 2, 'myVGG16')
test_model_and_submit('testing4', model, 1, 2, 'myVGG16')
