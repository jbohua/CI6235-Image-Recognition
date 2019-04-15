import os
from PIL import Image, ImageEnhance
from sklearn import model_selection, preprocessing

import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import numpy as np

original_dir = '.\\original'
img_dir = '.\\processed'
colors = ['red', 'green', 'yellow']
img_rows, img_cols = 64, 64
img_shape = (img_rows, img_cols)
num_classes = len(colors)

def save_img(img, count, img_dir_color):
    processed_path = os.path.join(img_dir_color, str(count) + '.jpg')
    img.save(processed_path)

def preprocess_images():

    for color in labels:

        original_dir_color = os.path.join(original_dir, color)
        img_dir_color = os.path.join(img_dir, color)

        if not os.path.exists(img_dir_color):
            os.makedirs(img_dir_color)

        count = 1
        for file in os.listdir(original_dir_color):
            if count > 1000:
                break
            original_path = os.path.join(original_dir_color, file)

            try:
                img = Image.open(original_path)
            except IOError:
                print('cannot open ' + original_path)
                continue

            for angle in range(0, 60, 3):

                if count > 1000:
                    break
                rotated_img = img.rotate(angle)
                save_img(rotated_img.resize(img_shape), count, img_dir_color)
                count += 1

                if count > 1000:
                    break
                flipped_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
                save_img(flipped_img.resize(img_shape), count, img_dir_color)
                count += 1

                if count > 1000:
                    break
                enhanced_img = ImageEnhance.Brightness(flipped_img)
                flipped_img = enhanced_img.enhance(1.5)
                save_img(flipped_img.resize(img_shape), count, img_dir_color)
                count += 1

        print('color - ' + color + ': images - ' + str(count - 1))

def load_img_array():
    dataset = []
    labels = []

    for color in colors:
        img_dir_color = os.path.join(img_dir, color)
        
        for file in os.listdir(img_dir_color):
            labels.append(color)
            
            file_path = os.path.join(img_dir_color, file)
            img = load_img(file_path)
            dataset.append(img_to_array(img))
    
    return np.array(dataset), np.array(labels)

def get_train_test_data():
    dataset, labels = load_img_array()
    
    encoder = preprocessing.LabelEncoder()
    encoder.fit(colors)
    labels = encoder.transform(labels)

    x_train, x_test, y_train, y_test= model_selection.train_test_split(dataset, labels, test_size=0.2, random_state=42, shuffle=True)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], num_classes, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], num_classes, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_classes)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_classes)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

def no_hidden_layer(x_train, x_test, y_train, y_test):
    
    input_shape=x_train[0].shape
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', input_shape=input_shape))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

    model.fit(x_train / 255, y_train,
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_data=(x_test / 255, y_test))
    
    score = model.evaluate(x_test / 255, y_test, verbose=0)
    print('no_hidden_layer')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('no_hidden_layer.h5')

def one_hidden_layer(x_train, x_test, y_train, y_test):
    input_shape=x_train[0].shape
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

    model.fit(x_train / 255, y_train,
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_data=(x_test / 255, y_test))
    
    score = model.evaluate(x_test / 255, y_test, verbose=0)
    print('one_hidden_layer')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('one_hidden_layer.h5')

def multiple_hidden_layers(x_train, x_test, y_train, y_test):
    
    input_shape=x_train[0].shape
    
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

    model.fit(x_train / 255, y_train,
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_data=(x_test / 255, y_test))
    
    score = model.evaluate(x_test / 255, y_test, verbose=0)
    print('with_hidden_layer')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('multiple_hidden_layers.h5')

if __name__ == '__main__':
    #preprocess_images()
    x_train, x_test, y_train, y_test = get_train_test_data()
    no_hidden_layer(x_train, x_test, y_train, y_test)
    one_hidden_layer(x_train, x_test, y_train, y_test)
    multiple_hidden_layers(x_train, x_test, y_train, y_test)