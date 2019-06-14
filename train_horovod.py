# Copyright 2017 onwards, fast.ai, Inc.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import datetime
import h5py
import io
import os
import numpy as np
import pyarrow as pa


from packaging import version
if version.parse(tf.__version__) < version.parse('1.5.0'):
	import tensorflow.python.keras.backend as K
	from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
	from tensorflow.python.keras.optimizers import Adam
else:
	import tensorflow.keras.backend as K
	from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
	from tensorflow.keras.optimizers import Adam

from train_model import unet
from pyspark import SparkConf
from pyspark.sql import SparkSession
import horovod.spark
import horovod.tensorflow.keras as hvd
from petastorm.codecs import NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField

# Location of data on local filesystem (prefixed with file://) or on HDFS.
DATA_LOCATION = 'hdfs://node013.ib.cluster:8020'
LOCAL_CHECKPOINT_FILE = 'checkpoint.h5'
DEFAULT_IMAGE_SIZE = (128, 128)
NUM_TRAINING_PROC = 49

# Desired sampling rate.  Useful to set to low number (e.g. 0.01) to make sure
# that end-to-end process works.
SAMPLE_RATE = None  # or use 0.01

# Batch size & learning rate to use.
BATCH_SIZE = 2
LR = 1e-4

# HDFS driver to use with Petastorm.
PETASTORM_HDFS_DRIVER = 'libhdfs'

# ============== #
# MODEL TRAINING #
# ============== #

print('==============')
print('Model training')
print('==============')

TrainSchema = [
    UnischemaField('features', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3), NdarrayCodec(), False),
    UnischemaField('masks', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), NdarrayCodec(), False)
]


def serialize_model(model):
    """Serialize model into byte array."""
    bio = io.BytesIO()
    with h5py.File(bio) as f:
        model.save(f)
    return bio.getvalue()


def deserialize_model(model_bytes, load_model_fn):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio) as f:
        return load_model_fn(f)

def decode_image(tensor):
        codec = NdarrayCodec()
        image_np_array = codec.decode(TrainSchema[0], tensor[0]).reshape((128, 128, 3))
        #print(image_np_array)
        return image_np_array

        #print("TYpe fffff  ========", (codec.decode(TrainSchema[0], tensor.features)))
        #return (codec.decode(TrainSchema[0], tensor[0]), codec.decode(TrainSchema[1], tensor[1]))

def decode_mask(tensor):
        codec = NdarrayCodec()
        mask_np_array = codec.decode(TrainSchema[1], tensor[1]).reshape(128, 128, 1)
        return mask_np_array

# use GPU for the session creation.
# config = tf.ConfigProto(device_count={'GPU': 0})
# K.set_session(tf.Session(config=config))

#DEFAULT_EPOCHS = 30

TRAIN_ROWS = 120
VAL_ROWS = 120

model = unet(pretrained_weights=None, input_size=(*DEFAULT_IMAGE_SIZE, 3))
opt = Adam(lr = 1e-4)
opt = hvd.DistributedOptimizer(opt)

model.compile(loss = 'binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
model_bytes = serialize_model(model)


def train_fn(model_bytes):
    # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
    # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
    # functions like `deserialize_model` which are implemented at the top level.
    # See https://jira.apache.org/jira/browse/ARROW-3346
    pa

    import atexit
    import horovod.tensorflow.keras as hvd
    import os
    from petastorm import make_batch_reader
    from petastorm.tf_utils import make_petastorm_dataset
    import tempfile
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import shutil

    # Horovod: initialize Horovod inside the trainer.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0, 
                        allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    # Horovod: restore from checkpoint, use hvd.load_model under the hood.
    model = deserialize_model(model_bytes, hvd.load_model)

    # Horovod: adjust learning rate based on number of processes.
    K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * hvd.size())

    # Horovod: print summary logs on the first worker.
    verbose = 2 if hvd.rank() == 0 or hvd.rank() == 1 else 0

    callbacks = [
        # # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # # This is necessary to ensure consistent initialization of all workers when
        # # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

        # # Horovod: average metrics among workers at the end of every epoch.
        # #
        # # Note: This callback must be in the list before the ReduceLROnPlateau,
        # # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose),

        # # Reduce LR if the metric is not improved for 10 epochs, and stop training
        # # if it has not improved for 20 epochs.
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_exp_rmspe', patience=10, verbose=verbose),
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        #tf.keras.callbacks.EarlyStopping(monitor='val_exp_rmspe', mode='min', patience=20, verbose=verbose),
        #tf.keras.callbacks.TerminateOnNaN()
    ]

    # Model checkpoint location.
    ckpt_dir = tempfile.mkdtemp()
    ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint(ckpt_dir, monitor='loss',
                                        verbose=1, save_best_only=True))

    # Make Petastorm readers.
    with make_batch_reader('%s/train/train_df.parquet' % DATA_LOCATION, num_epochs=None,
                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                           hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
        with make_batch_reader('%s/validation/validation_df.parquet' % DATA_LOCATION, num_epochs=None,
                               cur_shard=hvd.rank(), shard_count=hvd.size(),
                               hdfs_driver=PETASTORM_HDFS_DRIVER) as val_reader:
            # Convert readers to tf.data.Dataset.
            train_ds = make_petastorm_dataset(train_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .shuffle(int(TRAIN_ROWS / hvd.size())) \
                .map(lambda tensor: (tf.py_func(decode_image, [tensor], tf.uint8), tf.py_func(decode_mask, [tensor], tf.uint8))) \
                .batch(BATCH_SIZE) 
            #tf.print(tf.shape(train_ds))
            #iterator = train_ds.make_one_shot_iterator()
            #tensor = iterator.get_next()
            #with tf.Session() as sess:
              #sample = sess.run(tensor)
              #print(sample)

            val_ds = make_petastorm_dataset(val_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .map(lambda tensor: (tf.py_func(decode_image, [tensor], tf.uint8), tf.py_func(decode_mask, [tensor], tf.uint8))) \
                .batch(BATCH_SIZE) 

            history = model.fit(train_ds,
                                validation_data=val_ds,
                                steps_per_epoch=int(TRAIN_ROWS / BATCH_SIZE / hvd.size()),
                                validation_steps=int(VAL_ROWS / BATCH_SIZE / hvd.size()),
                                callbacks=callbacks,
                                verbose=verbose,
                                epochs=2)

    # Dataset API usage currently displays a wall of errors upon termination.
    # This global model registration ensures clean termination.
    # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
    globals()['_DATASET_FINALIZATION_HACK'] = model

    if hvd.rank() == 0:
        with open(ckpt_file, 'rb') as f:
            return history.history, f.read()


# Create Spark session for training.
conf = SparkConf().setAppName('training')
#if TRAINING_CLUSTER:
    #conf.setMaster(TRAINING_CLUSTER)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Horovod: run training.
history, best_model_bytes = \
    horovod.spark.run(train_fn, args=(model_bytes,), num_proc=NUM_TRAINING_PROC, verbose=2)[0]

best_val_lost = min(history['val_loss'])
print('Best loss: %f' % best_val_lost)

# Write checkpoint.
with open(LOCAL_CHECKPOINT_FILE, 'wb') as f:
    f.write(best_model_bytes)
print('Written checkpoint to %s' % LOCAL_CHECKPOINT_FILE)

spark.stop()

