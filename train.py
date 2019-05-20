import os
from train_model import unet
from augmentation.data_generator import DataGenerator

import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

if __name__=="__main__":
    # Parameters
    IMAGE_SIZE = 4096, 4096
    BATCH_SIZE = 1

    train_model = unet(pretrained_weights=None, input_size=(*IMAGE_SIZE, 3))
    train_model.compile(loss = 'binary_crossentropy',
                        optimizer=Adam(lr = 1e-4),
                        metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',
                                        verbose=1, save_best_only=True)
    train_dir = os.getcwd() + '/train_data'
    print("Train dir ->> ", train_dir)
    train_images = os.listdir(train_dir + '/images')
    mask_images = os.listdir(train_dir + '/masks')

    print("Train images =>>>>>")
    print(train_images)
    print("====================")

    data_generator = DataGenerator(image_folder=train_dir, images=train_images, masks=mask_images,
                                    batch_size=BATCH_SIZE, dim=IMAGE_SIZE, nchannels=3)
    train_model.fit_generator(data_generator, 
                        steps_per_epoch=int(len(train_images)/BATCH_SIZE) , epochs=5,
                        callbacks=[model_checkpoint])


