import os
from train_model import unet
from augmentation.data_generator import DataGenerator

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import tensorflow.python.keras
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


if __name__=="__main__":
    # Parameters
    IMAGE_SIZE = 256, 256
    BATCH_SIZE = 10
    DEFAULT_EPOCHS = 10

    train_model = unet(pretrained_weights=None, input_size=(*IMAGE_SIZE, 3))
    train_model.compile(loss = 'binary_crossentropy',
                        optimizer=Adam(lr = 1e-4),
                        metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('unet_building.hdf5', monitor='loss',
                                        verbose=1, save_best_only=True)
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    train_dir = os.getcwd() + '/train_data'
    train_images = os.listdir(train_dir + '/images')
    train_mask_images = os.listdir(train_dir + '/masks')
   
    validation_dir = os.getcwd() + '/validation_data'
    validation_images = os.listdir(validation_dir + '/images')
    validation_mask_images = os.listdir(validation_dir + '/masks')
    print("Start training =>>>>>")
    print("Train images: {}; Train masks: {} ; Validation images {} ; Validation masks: {}" \
    .format(len(train_images), len(train_mask_images), len(validation_images), len(validation_mask_images))) 
    print("====================")

    train_generator = DataGenerator(image_folder=train_dir, images=train_images, masks=train_mask_images,
                                    batch_size=BATCH_SIZE, dim=IMAGE_SIZE, nchannels=3)
    validation_generator = DataGenerator(image_folder=validation_dir, images=validation_images, masks=validation_mask_images,
                                    batch_size=BATCH_SIZE, dim=IMAGE_SIZE, nchannels=3)
    train_model.fit_generator(train_generator, 
                        steps_per_epoch=int(len(train_images)/BATCH_SIZE),
                        validation_data=validation_generator,
                        validation_steps=int(len(validation_images)/BATCH_SIZE),
                        epochs=DEFAULT_EPOCHS,
                        callbacks=[model_checkpoint, es])



