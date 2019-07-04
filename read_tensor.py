import tensorflow as tf
from petastorm import make_batch_reader
from petastorm import make_reader
from petastorm.tf_utils import make_petastorm_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
import numpy as np
import tensorflow.keras.backend as K
import shutil
import horovod.tensorflow.keras as hvd
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
tf.enable_eager_execution()
DEFAULT_IMAGE_SIZE = (128, 128)

TrainSchema = [
    UnischemaField('features', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3), NdarrayCodec(), False),
    UnischemaField('masks', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), NdarrayCodec(), False)
]

    # Horovod: initialize Horovod inside the trainer.
hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
from petastorm.transform import TransformSpec
 

def decode_image(tensor):
	codec = NdarrayCodec()
	image_np_array = codec.decode(TrainSchema[0], tensor[0])
	print(image_np_array.shape)
	return image_np_array

	#print("TYpe fffff  ========", (codec.decode(TrainSchema[0], tensor.features)))
	#return (codec.decode(TrainSchema[0], tensor[0]), codec.decode(TrainSchema[1], tensor[1]))

def decode_mask(tensor):
	codec = NdarrayCodec()
	mask_np_array = codec.decode(TrainSchema[1], tensor[1])
	return mask_np_array
 
#transform = TransformSpec(decode_image_and_mask)
with make_batch_reader('hdfs://node013.ib.cluster:8020/train/train_df.parquet', num_epochs=None,
                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                           hdfs_driver='libhdfs') as train_reader:
	train_ds = make_petastorm_dataset(train_reader) \
					.apply(tf.data.experimental.unbatch())\
					.map(lambda tensor: (tf.py_func(decode_image, [tensor], tf.uint8), tf.py_func(decode_mask, [tensor], tf.uint8)))
	iterator = train_ds.make_one_shot_iterator()
	for x, y in iterator:
		print(x)
		
	#tensor = iterator.get_next()
	#with tf.Session() as sess:
		#print("v1", tf.shape(tensor))
		#sample = sess.run(tf.shape(tensor))
		#print(sample)
		#next_element = iterator.get_next()
			#sess.run(iterator.initializer)
			#while True:
			#	try:
			#		(features, labels) = sess.run(tensor)
			#		print("Features ->>>")
			#		#print(features.shape)
			#		print("=============")
			#	except tf.errors.OutOfRangeError:
		#		print("end of training dataset")	
