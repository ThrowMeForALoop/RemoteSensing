import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
import numpy as np
from PIL import Image
from io import BytesIO
from pyspark.ml.image import ImageSchema
from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import DenseVector  

from pyspark.sql.types import _infer_schema
from pyspark.sql.functions import monotonically_increasing_id


# Load spark context locally
sc = SparkContext("local[*]", "Images To Parquet")
images_rdd = sc.binaryFiles("file:///Volumes/Data/VUSubjects/Thesis/src/Test/data/tiff_images/*")
masks_rdd = sc.binaryFiles("file:///Volumes/Data/VUSubjects/Thesis/src/Test/data/masks/*")

# Decode binary to image and transform image into numpy array
image_to_array = lambda rawdata: np.asarray(Image.open(BytesIO(rawdata))).reshape((-1, ))

image_flat_numpy_rdd = images_rdd.values().map(image_to_array).map(lambda x: (DenseVector(x),))
masks_flat_numpy_rdd = masks_rdd.values().map(image_to_array)\
                                         .map(lambda array: array / 255 ) \
                                         .map(lambda x: (DenseVector(x),))

session = SparkSession(sc)
# test_array = DenseVector(np.arange(10))
# print(_infer_schema((test_array, )))


images_df = image_flat_numpy_rdd.toDF(["features"]).withColumn("id", monotonically_increasing_id())
mask_df = masks_flat_numpy_rdd.toDF(["masks"]).withColumn("id", monotonically_increasing_id())

train_df = images_df.join(mask_df, "id", "outer").drop("id")
# df.show()
#df.write.parquet("file:///output/")
# df.createOrReplaceTempView("image_numpy_rdd")
train_df.show()

train_df.write.mode('overwrite').parquet("file:///Volumes/Data/VUSubjects/Thesis/src/Test/data/output/train.parquet")

# for r in image_numpy_rdd.collect():
#     print(r.shape)

# sc.stop()

print("Done")