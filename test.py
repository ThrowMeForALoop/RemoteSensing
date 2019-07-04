import os
from augmentation.data_generator import DataGenerator

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

import numpy as np
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

if __name__=="__main__":
    # Parameters
    IMAGE_SIZE = 128, 128
    BATCH_SIZE = 1
    DEFAULT_EPOCHS = 10

    
    test_dir = os.getcwd() + '/test_data'
    test_images_dir = test_dir + '/images'
    test_mask_images_dir = test_dir + '/masks'
   
    test_images = os.listdir(test_images_dir)
    test_masks = os.listdir(test_mask_images_dir)

    print("Start testing =>>>>>")
    print("Test images: {}; Test masks: {}" \
    .format(len(test_images), len(test_masks)))
    print("====================")

    # Load model from hdf5 file
    model = load_model('unet_building128_16b_30e.hdf5')
    print(model.summary())
    
    # Load test data
    test_generator = DataGenerator(test_dir, test_images, None,
                batch_size=1, dim=(128, 128), nchannels=3, shuffle=False)
    predict = model.predict_generator(test_generator, steps = len(test_images))
    predict = ((predict > 0.3).astype(np.uint8)) * 255
    print("Shape of predict", predict.shape)
    for i,item in enumerate(predict):
        img = Image.fromarray(item[:,:,0])
        img_name = test_images[i].replace(".png", "")
        img.save(os.path.join(os.getcwd(), "result", "%s_predict.png"%img_name))
        print("Complete save predict image {}".format(img_name))
