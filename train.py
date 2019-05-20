import datetime
import h5py
import io
import os
#import pyarrow as pa

from pyspark import SparkConf, Row
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

import tensorflow as tf
import tensorflow.keras.backend as K
import horovod.spark
import horovod.tensorflow as hvd

print("I am here")
def unet(pretrained_weights = None,input_size = (256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
    
def serialize_model(model):
    """Serialize model into byte array."""
    bio = io.BytesIO()
    with h5py.File(bio) as f:
        model.save(f)
    return bio.getvalue()

# CUSTOM_OBJECTS = {'exp_rmspe': exp_rmspe,
#                   'act_sigmoid_scaled': act_sigmoid_scaled}

def deserialize_model(model_bytes, load_model_fn):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio) as f:
        return load_model_fn(f)

DATA_LOCATION = 'hdfs:///output'
PETASTORM_HDFS_DRIVER = 'libhdfs'

def train_fn(model_bytes):
    # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
    # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
    # functions like `deserialize_model` which are implemented at the top level.
    # See https://jira.apache.org/jira/browse/ARROW-3346
    #pa 
     
    train_rows = 2
    BATCH_SIZE = 1

    import atexit
    import horovod.tensorflow.keras as hvd
    import os
    from petastorm import make_batch_reader
    from petastorm.tf_utils import make_petastorm_dataset
    import tempfile
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import shutil
    # import pyarrow as pa
    print("=>>>>>>>>>> Start init enviroment")
    # Horovod: initialize Horovod inside the trainer.
    hvd.init()

    print("=>>>>>>>>>> Start get gpu")
    # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    print("=>>>>>>>>>> Get Model")
    # Horovod: restore from checkpoint, use hvd.load_model under the hood.
    model = deserialize_model(model_bytes, hvd.load_model)

    # Horovod: adjust learning rate based on number of processes.
    K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * hvd.size())

    # Horovod: print summary logs on the first worker.
    verbose = 2 if hvd.rank() == 0 else 0

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose),

        # Reduce LR if the metric is not improved for 10 epochs, and stop training
        # if it has not improved for 20 epochs.
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=verbose),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=verbose),
        
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # Model checkpoint location.
    ckpt_dir = tempfile.mkdtemp()
    ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    print("=>>>>>>>>>> Upload model file") 

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='val_loss', mode='min',
                                                            save_best_only=True))

    print("=>>>>>>>>>> Start reading data")
    # Make Petastorm readers.
    with make_batch_reader('%s/train.parquet' % DATA_LOCATION, num_epochs=None,
                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                           hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
        # with make_batch_reader('%s/val_df.parquet' % DATA_LOCATION, num_epochs=None,
        #                        cur_shard=hvd.rank(), shard_count=hvd.size(),
        #                        hdfs_driver=PETASTORM_HDFS_DRIVER) as val_reader:
            # Convert readers to tf.data.Dataset.
         
            train_ds = make_petastorm_dataset(train_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .shuffle(int(train_rows / hvd.size())) \
                .batch(BATCH_SIZE) \
                .map(lambda x: (x.features.reshape(5000, 5000, 3), x.masks.reshape(5000, 5000, 1)))

            # val_ds = make_petastorm_dataset(val_reader) \
            #     .apply(tf.data.experimental.unbatch()) \
            #     .batch(BATCH_SIZE) \
            #     .map(lambda x: (tuple(getattr(x, col) for col in all_cols), tf.log(x.Sales)))

            history = model.fit(train_ds,
                                steps_per_epoch=int(train_rows / BATCH_SIZE / hvd.size()),     
                                callbacks=callbacks,
                                verbose=verbose,
                                epochs=100)
                                # validation_data=val_ds,
                                # validation_steps=int(val_rows / BATCH_SIZE / hvd.size()),)

    # Dataset API usage currently displays a wall of errors upon termination.
    # This global model registration ensures clean termination.
    # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
    globals()['_DATASET_FINALIZATION_HACK'] = model

    if hvd.rank() == 0:
        with open(ckpt_file, 'rb') as f:
            return history.history, f.read()


if __name__ == "__main__":
    import sys
    im_height = 4096
    im_width = 4096
    band = 3
    NUM_TRAINING_PROC = 4
    LOCAL_CHECKPOINT_FILE = 'checkpoint.h5'

    print("DATA LOCATIOn ->>>>", DATA_LOCATION)
    # Construct first layer
    input_img = Input((im_height, im_width, band), name='img')

    # ===== Prepare model ===== 
    # Do not use GPU for the session creation.
    config = tf.ConfigProto(device_count={'GPU': 0})
    K.set_session(tf.Session(config=config))

    # Build the model.
    model = unet(pretrained_weights=None, input_size=(im_height, im_width, band))
    model.summary()

    # Horovod: add Distributed Optimizer.
    opt = Adam(lr = 1e-4)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(optimizer = opt, loss='binary_crossentropy', metrics = ['accuracy'])
    model_bytes = serialize_model(model)
    
    # ===== Start training ====
    # Create Spark session for training.
    conf = SparkConf().setAppName('training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Horovod: run training.
    history, best_model_bytes = \
        horovod.spark.run(train_fn, args=(model_bytes,), num_proc=NUM_TRAINING_PROC, verbose=2)[0]

    best_val_loss = min(history['val_loss'])
    print('Best loss: %f' % best_val_loss)

    # Write checkpoint.
    with open(LOCAL_CHECKPOINT_FILE, 'wb') as f:
        f.write(best_model_bytes)
    print('Written checkpoint to %s' % LOCAL_CHECKPOINT_FILE)

    spark.stop()
