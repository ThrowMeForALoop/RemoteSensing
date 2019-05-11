import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
import numpy as np
from PIL import Image
from io import BytesIO
from pyspark.ml.image import ImageSchema
from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import DenseVector  

from pyspark.sql.types import _infer_schema
from pyspark.sql.functions import monotonically_increasing_id

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputpath", help="image input path")
parser.add_argument("--outputpath", help="parquet output path")


args = parser.parse_args()

input_path = None
output_path = None

if args.inputpath:
    input_path = args.inputpath
if args.outputpath:
    output_path = args.outputpath

if input_path is None or output_path is None:
    print('<Usage> --input inputpath  --output outputpath')
    sys.exit() 

# Load spark context
spark_conf = SparkConf().setAppName('Image preprocessing')
sc = SparkContext(conf=spark_conf)

# Load input data
images_rdd = sc.binaryFiles(input_path + 'tiff_images/*')
masks_rdd = sc.binaryFiles(input_path + 'masks/*')

# Decode binary to image and transform image into numpy array
image_to_array = lambda rawdata: np.asarray(Image.open(BytesIO(rawdata))).reshape((-1, ))

image_flat_numpy_rdd = images_rdd.values().map(image_to_array).map(lambda x: (DenseVector(x),))
masks_flat_numpy_rdd = masks_rdd.values().map(image_to_array)\
                                         .map(lambda array: array / 255 ) \
                                         .map(lambda x: (DenseVector(x),))

# Create session to create data frame
session = SparkSession(sc)


images_df = image_flat_numpy_rdd.toDF(["features"]).withColumn("id", monotonically_increasing_id())
mask_df = masks_flat_numpy_rdd.toDF(["masks"]).withColumn("id", monotonically_increasing_id())

train_df = images_df.join(mask_df, "id", "outer").drop("id")
train_df.repartition(1).write.mode('overwrite') \
	.parquet("file:///var/scratch/tnguyenh/src/RemoteSensing/output/train.parquet")

# for r in image_numpy_rdd.collect():
#     print(r.shape)

sc.stop()

print("========Done=========")
