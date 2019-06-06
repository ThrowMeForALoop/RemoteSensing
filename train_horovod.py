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
import pyarrow as pa

import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import Adam
from train_model import unet

import horovod.spark
import horovod.tensorflow.keras as hvd

# Location of data on local filesystem (prefixed with file://) or on HDFS.
DATA_LOCATION = 'hdfs://input'
LOCAL_CHECKPOINT_FILE = 'checkpoint.h5'

# Spark clusters to use for training. If set to None, uses current default cluster.
#
# Light processing (data preparation & prediction) uses typical Spark setup of one
# task per CPU core.
#
# Training cluster should be set up to provide a Spark task per multiple CPU cores,
# or per GPU, e.g. by supplying `-c <NUM GPUs>` in Spark Standalone mode.

TRAINING_CLUSTER = None  # or 'spark://hostname:7077'

# The number of training processes.
NUM_TRAINING_PROC = 8

# Desired sampling rate.  Useful to set to low number (e.g. 0.01) to make sure
# that end-to-end process works.
SAMPLE_RATE = None  # or use 0.01

# Batch size & learning rate to use.
BATCH_SIZE = 10
LR = 1e-4

# HDFS driver to use with Petastorm.
PETASTORM_HDFS_DRIVER = 'libhdfs'

# ============== #
# MODEL TRAINING #
# ============== #

print('==============')
print('Model training')
print('==============')

# from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Flatten, Reshape, BatchNormalization, Dropout
# import tensorflow.keras.backend as K
# import horovod.spark
# import horovod.tensorflow.keras as hvd


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


# # Do not use GPU for the session creation.
# config = tf.ConfigProto(device_count={'GPU': 0})
# K.set_session(tf.Session(config=config))

# # Build the model.
# inputs = {col: Input(shape=(1,), name=col) for col in all_cols}
# embeddings = [Embedding(len(vocab[col]), 10, input_length=1, name='emb_' + col)(inputs[col])
#               for col in categorical_cols]
# continuous_bn = Concatenate()([Reshape((1, 1), name='reshape_' + col)(inputs[col])
#                                for col in continuous_cols])
# continuous_bn = BatchNormalization()(continuous_bn)
# x = Concatenate()(embeddings + [continuous_bn])
# x = Flatten()(x)
# x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
# x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
# x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
# x = Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
# x = Dropout(0.5)(x)
# output = Dense(1, activation=act_sigmoid_scaled)(x)
# model = tf.keras.Model([inputs[f] for f in all_cols], output)
# model.summary()
 # Parameters
IMAGE_SIZE = 128, 128
BATCH_SIZE = 16
DEFAULT_EPOCHS = 30

TRAIN_ROWS = 10000
VAL_ROWS = 2000

model = unet(pretrained_weights=None, input_size=(*IMAGE_SIZE, 3))
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    # Horovod: restore from checkpoint, use hvd.load_model under the hood.
    model = deserialize_model(model_bytes, hvd.load_model)

    # Horovod: adjust learning rate based on number of processes.
    K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * hvd.size())

    # Horovod: print summary logs on the first worker.
    verbose = 2 if hvd.rank() == 0 else 0

    callbacks = [
        # # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # # This is necessary to ensure consistent initialization of all workers when
        # # training is started with random weights or restored from a checkpoint.
        # hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

        # # Horovod: average metrics among workers at the end of every epoch.
        # #
        # # Note: This callback must be in the list before the ReduceLROnPlateau,
        # # TensorBoard, or other metrics-based callbacks.
        # hvd.callbacks.MetricAverageCallback(),

        # # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose),

        # # Reduce LR if the metric is not improved for 10 epochs, and stop training
        # # if it has not improved for 20 epochs.
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_exp_rmspe', patience=10, verbose=verbose),
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # tf.keras.callbacks.EarlyStopping(monitor='val_exp_rmspe', mode='min', patience=20, verbose=verbose),
        # tf.keras.callbacks.TerminateOnNaN()
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
    with make_batch_reader('%s/train_df.parquet' % DATA_LOCATION, num_epochs=None,
                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                           hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
        with make_batch_reader('%s/val_df.parquet' % DATA_LOCATION, num_epochs=None,
                               cur_shard=hvd.rank(), shard_count=hvd.size(),
                               hdfs_driver=PETASTORM_HDFS_DRIVER) as val_reader:
            # Convert readers to tf.data.Dataset.
            train_ds = make_petastorm_dataset(train_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .shuffle(int(TRAIN_ROWS / hvd.size())) \
                .batch(BATCH_SIZE) \
                .map(lambda x: (x.features, x.masks))

            val_ds = make_petastorm_dataset(val_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .batch(BATCH_SIZE) \
                .map(lambda x: (x.features, x.masks))

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
if TRAINING_CLUSTER:
    conf.setMaster(TRAINING_CLUSTER)
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

