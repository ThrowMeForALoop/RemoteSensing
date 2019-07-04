import os
import numpy as np
from tensorflow.python import keras 
import math
from PIL import Image

# TODO: User properties for validating parameters
class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_folder, images, masks, 
                batch_size=10, dim=(256, 256), nchannels=3,
                shuffle=True):
        self.image_folder = image_folder
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.dim = dim
        self.nchannels = nchannels
        self.shuffle = shuffle
        self.indexes = None
        # Init indexes with random manner
        self.on_epoch_end()

    def __len__(self):
        """
        Lenth of a batch
        """
        return int(math.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """[summary]
        Generate one batch of data
        Arguments:
            index {[int]} -- index of batch
        """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_images = [self.images[k] for k in indexes]

        # Generate batch data
        return self.__data__generation(batch_images)

    def __data__generation(self, batch_images):
        """[summary]
            generate tuple of numpy array of training input and mask
        Arguments:
            batch_images {list of images in each batch}
        """

        X = np.empty((self.batch_size, *self.dim, self.nchannels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        for index, image in enumerate(batch_images):
            train_img = Image.open(self.image_folder + '/images/' + image) 
            X[index, ] = np.asarray(train_img).astype('float32') / 255
            
            if self.masks is None:
                return X

            mask_img = Image.open(self.image_folder + '/masks/' + image)  
            Y[index, ] = np.asarray(mask_img).reshape((*self.dim, 1)).astype('float32') / 255
        return X, Y

    def on_epoch_end(self):
        """[summary]
        Updates indexes after each epoch
        """
        print("Epoch end =>>>")
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) 
    
