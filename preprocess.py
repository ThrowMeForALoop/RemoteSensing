import numpy as np
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from pyspark.sql.types import IntegerType
from PIL import Image
from io import BytesIO

DEFAULT_IMAGE_SIZE = (128, 128)

# The schema defines how the dataset schema looks like
FeatureSchema = Unischema('FeatureSchema', [
    UnischemaField('features', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3), CompressedImageCodec('png'), False),
    UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
])

MaskSchema = Unischema('MaskSchema', [
    UnischemaField('masks', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), CompressedImageCodec('png'), False),
    UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
])

TrainSchema = Unischema('TrainSchema', [
    UnischemaField('features', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3), CompressedImageCodec('png'), False),
    UnischemaField('masks', np.uint8, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), CompressedImageCodec('png'), False)
])

# def resize_image(raw_image_data, image_size = (128, 128)):
#     img = Image.open(BytesIO(raw_image_data))
#     img = img.resize((image_size[0], image_size[1]), Image.ANTIALIAS)
#     return img

def raw_image_to_numpy_array(raw_image_data):
    img = Image.open(BytesIO(raw_image_data))
    return np.asarray(img)

def generate_parquet(feature_path, mask_path, output_path):
    """[summary]
    Generate parquet file with two columns
        - First column: npG_array representing image
        - Second column: np_array representing mask

    Arguments:
        feature_path {[type]} -- path to all images
        mask_path {[type]} -- path to masks of images
        output_path {[type]} -- parquet path
    """

    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession, Row
    
    
    from pyspark.sql import Row
    from pyspark.sql.types import _infer_schema
    from pyspark.sql.functions import monotonically_increasing_id

    rowgroup_size_mb = 256
    spark_conf = SparkConf().setAppName('Image preprocess')
    sc = SparkContext(conf=spark_conf)
    session = SparkSession(sc)

    # Load images and convert it to dataframe
    images_rdd = sc.binaryFiles(feature_path).values()
    image_flat_numpy_rdd = images_rdd.map(lambda pair_raw_image_id: (raw_image_to_numpy_array(pair_raw_image_id[0]), pair_raw_image_id[1])) \
                                     .map(lambda pair_np_array_id: {'features': pair_np_array_id[0], 'id': pair_np_array_id[1]}) \
                                     .map(lambda x: dict_to_spark_row(FeatureSchema, x))

    image_df = session.createDataFrame(image_flat_numpy_rdd, FeatureSchema.as_spark_schema())
                       # .withColumn("id", monotonically_increasing_id()) # Generate table row id 
    # Load masks and convert it to dataframe
    mask_rdd = sc.binaryFiles(mask_path).values().zipWithIndex()

    # Convert mask rgb value to 0 for not building and 1 for building
    mask_flat_numpy_rdd = mask_rdd.map(lambda pair_raw_image_id: (raw_image_to_numpy_array(pair_raw_image_id[0]), pair_raw_image_id[1])) \
                                           .map(lambda pair_np_array_id: ((pair_np_array_id[0] / 255).astype(np.uint8), pair_np_array_id[1])) \
                                           .map(lambda pair_std_np_array_id: {'masks': pair_std_np_array_id[0], 'id': pair_std_np_array_id[1]}) \
                                           .map(lambda x: dict_to_spark_row(MaskSchema, x))

    mask_df = session.createDataFrame(mask_flat_numpy_rdd, MaskSchema.as_spark_schema()) 
                        #.withColumn("id", monotonically_increasing_id()) # Generate table row id 
    mask_df.show(5, False)
    # Concat image_df and mask_df row by row
    train_df = image_df.join(mask_df, "id", "inner").drop('id')
   
    #print("Summary =>>>>>>>>>>>>>>>>>>>>>>>....>>>")
    #print("Image count {} , mask count {}, train_count {}".format(image_df.count(), mask_df.count(), train_df.count()))
    #print("=======================================")
    with materialize_dataset(session, output_path, TrainSchema, rowgroup_size_mb):
           train_df.write \
                    .mode('overwrite') \
                    .parquet(output_path)


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--featurepath", help="Path to train images")
    parser.add_argument("--maskpath", help="Path to train masks")
    parser.add_argument("--outputpath", help="Path to output parquet")

    args = parser.parse_args()
    feature_path = args.featurepath
    mask_path = args.maskpath
    output_path = args.outputpath

    if feature_path == None or mask_path == None or output_path == None:
        print('<Usage> --featurepath /path to train images --maskpath /path to train masks --outputpath /path to output parquet')
        print('===== Without parameters referring to hdfs, preprocessing will run locally =======')
        
        # TODO: get pwd 
        feature_path = 'file:///Volumes/Data/VUSubjects/Thesis/src/Test/data/input/features/*.tif'
        mask_path = 'file:///Volumes/Data/VUSubjects/Thesis/src/Test/data/input/masks/*.tif'
        output_path = 'file:///Volumes/Data/VUSubjects/Thesis/src/Test/data/output/train'

    generate_parquet(feature_path=feature_path, 
                     mask_path=mask_path,
                     output_path=output_path)

    
    
