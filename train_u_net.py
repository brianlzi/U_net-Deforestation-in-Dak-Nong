import tensorflow as tf
import datetime
import tifffile as tiff
import numpy as np
import random
import os


from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU, MaxPool2D, Input, UpSampling2D
from tensorflow.keras import models
from tensorflow.keras import layers
from keras import backend as K
from keras.backend import categorical_crossentropy
from keras.utils import np_utils
from tensorflow.keras.models import model_from_json
import keras

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
session = tf.Session(config=config)



height = 32
width = 32
channels = 13
num_label = 11
smooth = 1e-12



def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 2, 1])
    sum_ = K.sum(y_true, axis=[0, 2, 1])

    jac = (intersection + smooth) / (sum_ + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 2, 1])
    sum_ = K.sum(y_true , axis=[0, 2, 1])

    jac = (intersection + smooth) / (sum_ + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + categorical_crossentropy(y_pred, y_true)


def read_model(json_name, weight_name):
   json_name = "cache/" + json_name
   weight_name = "cache/" +  weight_name
   model = model_from_json(open(json_name).read())
   model.load_weights(weight_name)
   return model


def batch_generator(X_1, X_2, y, batch_size):
    while True:
        X_batch = np.zeros((batch_size, height, width, channels))
        y_batch = np.zeros((batch_size, height, width, 11))
       
        X_height = X_1.shape[1]
        X_width = X_1.shape[2]

        i = 0
        while i < batch_size:
            random_width = random.randint(0, X_width - width - 1)
            random_height = random.randint(0, X_height - height - 1)

            y_batch[i] = np.array(y[0, random_height: random_height + height, random_width: random_width + width, :])
            if np.sum(y_batch[i]) / (width * height) <= 0.2:  
                continue
            
            random_X = random.randint(0, 1)
            if random_X == 0:
                X_batch[i] = np.array(X_1[0, random_height: random_height + height, random_width: random_width + width, :])
            else:
                X_batch[i] = np.array(X_2[0, random_height: random_height + height, random_width: random_width + width, :])

            i += 1

        yield X_batch, y_batch


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def get_unet():
    '''
    model function for u_net
    '''
    #Input layer
    input_layer = Input((height, width, channels), batch_size= 512)
    
    #Conv 1 128
    conv1 = Conv2D(filters = 32, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ELU()(conv1)
    conv1 = Conv2D(filters = 32, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ELU()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    #Conv 2 64
    conv2 = Conv2D(filters = 64, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ELU()(conv2)
    conv2 = Conv2D(filters = 64, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ELU()(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    #Conv 3 32
    conv3 = Conv2D(filters = 128, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ELU()(conv3)
    conv3 = Conv2D(filters = 128, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ELU()(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    #Conv 4 16
    conv4 = Conv2D(filters = 256, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ELU()(conv4)
    conv4 = Conv2D(filters = 256, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ELU()(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    
    #Conv 5 8
    conv5 = Conv2D(filters = 512, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ELU()(conv5)
    conv5 = Conv2D(filters = 512, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ELU()(conv5)

    #upsample 6
    up6 =layers.concatenate([UpSampling2D(size=(2, 2))(conv5),conv4], 3)
    conv6 = Conv2D(filters = 256, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ELU()(conv6)
    conv6 = Conv2D(filters = 256, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ELU()(conv6)
    
    #upsample 7
    up7 =layers.concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], 3)
    conv7 = Conv2D(filters = 128, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ELU()(conv7)
    conv7 = Conv2D(filters = 128, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ELU()(conv7)

    #upsample 8
    up8 =layers.concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], 3)
    conv8 = Conv2D(filters = 64, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ELU()(conv8)
    conv8 = Conv2D(filters = 64, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ELU()(conv8)

    #upsample 9
    up9 =layers.concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], 3)
    conv9 = Conv2D(filters = 32, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ELU()(conv9)
    conv9 = Conv2D(filters = 32, kernel_size = 3, padding= "same", kernel_initializer='he_uniform')(conv9)
    # crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ELU()(conv9)

    #output
    conv10 = Conv2D(num_label, 1, 1, activation='softmax')(conv9)
    model = models.Model(inputs = input_layer, outputs = conv10)

    return model



if __name__ == '__main__':

    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet()

    print('[{}] Reading train.....................................'.format(str(datetime.datetime.now())))
    
    # x_train
    x_1 = tiff.imread("S2B_MSIL1C_20180207T030859.tif")
    x_1 = x_1/ (pow(2,16)-1)
    x_1 = np.transpose(x_1,[1,2,0])
    x_1 = np.expand_dims(x_1, 0)

    x_2 = tiff.imread("S2B_MSIL1C_20180217T030759.tif")
    x_2 = x_2/ (pow(2,16)-1)
    x_2 = np.transpose(x_2,[1,2,0])
    x_2 = np.expand_dims(x_2, 0)
    

    # y_train
    label = tiff.imread("label_PYU3.tif")
    train_label = np_utils.to_categorical(label)
    train_label = np.expand_dims(train_label, 0)
    print(train_label.shape)

    
    batch_size = 512
    suffix = "Class"

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss = jaccard_coef_loss , metrics=['categorical_crossentropy', jaccard_coef_int])
    now = datetime.datetime.now()
    model.fit_generator(batch_generator(x_1, x_2, train_label, batch_size),
                        epochs=100,
                        verbose=1,
                        steps_per_epoch= 400
                        )
    end = datetime.datetime.now()
    print("Finish in {}".format(end - now))
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=100, suffix=suffix))