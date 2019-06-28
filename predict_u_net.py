from tensorflow.keras.models import model_from_json
import tifffile as tiff
import numpy as np
import cv2
import datetime
from matplotlib import pyplot as plt



def merge_image(X, size, model):
    height = X.shape[1]
    width = X.shape[2]
    m = height % size
    n = width % size
    if height % size != 0:
        n_height = (height // size +1) * size
    if width % size != 0:
        n_width = (width // size + 1) * size

    num_img = (n_height // size) * (n_width // size)
    x_predict = np.zeros((num_img, size, size, channels))
    y_predict = np.zeros((num_img, size, size, num_label))

    i = 0
    for r in range(0, n_height, size):
        if (r + size) == n_height:
            tempt = X[:, r:r + size, c:c + size, :]
            x_predict[i, :tempt.shape[1], :tempt.shape[2], :] =  tempt
            i += 1
        else:
            for c in range(0, n_width , size):
                if (c + size) == n_width:
                    tempt = X[:, r:r + size, c:c + size, :]
                    x_predict[i, :tempt.shape[1], :tempt.shape[2], :] =  tempt
                    i += 1
                else:
                    x_predict[i] =  X[:, r:r + size, c:c + size, :]
                    i += 1
    
    y_predict = model.predict(x_predict)
    print("-------------------------------")
    print(x_predict.shape)
    y_merge = np.zeros((n_height, n_width))
    
    i = 0
    for r in range(0, height, size):
        for c in range(0, width , size):
            y_merge[r:r + size, c:c + size] = np.argmax(y_predict[i], axis = 2)
            i += 1
    y_merge = y_merge[:height, :width]
    return y_merge



def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d



def read_model(json_name, weight_name):
  json_name = "cache/" + json_name
  weight_name = "cache/" +  weight_name
  model = model_from_json(open(json_name).read())
  model.load_weights(weight_name)
  
  return model


if __name__=='__main__':

    start = datetime.datetime.now()

    size = 32
    channels = 13
    num_label = 11
    palette = {0 : (255, 255, 255),
           1 : (0, 0, 255),    
           2 : (0, 255, 255),  
           3 : (0, 255, 0),    
           4 : (255, 255, 0),  
           5 : (255, 0, 0),    
           6 : (0, 0, 0),  #61    
           7 : (255, 0, 255),  #62
           8 : (128, 0, 0),  #77
           9 : (0, 128, 128),  #78
           10 :(128, 128, 0)    #79
           }   



    x_train = tiff.imread("S2B_MSIL1C_20180207T030859.tif")
    x_train = x_train/ (pow(2,16)-1)
    x_train = np.transpose(x_train,[1,2,0])
    x_train = np.expand_dims(x_train, 0)
    print(x_train.shape)

    model = read_model(json_name = 'architecture_128_17000_Class_con.json', weight_name = 'model_weights_128_17000_Class_con.h5')
    print("READING DONE!")

    label = merge_image(x_train, size, model)
    arr = convert_to_color(label, palette = palette)

    print("Saving................")

    tiff.imsave("cache/test_0207_17000.tif",label)
    end = datetime.datetime.now()

    plt.imshow(arr)
    plt.show()  

    print("DONE in {}".format(end - start))

    