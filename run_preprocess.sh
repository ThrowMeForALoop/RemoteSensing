#!/bin/bash
PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs/bin/python spark-submit --master spark://node013.ib.cluster:7077 \
 								     --executor-memory 8G \
								     --num-executors 16 \
								     --total-executor-cores 49 \
								     preprocess.py \
								     --featurepath hdfs:///preprocess/train/images \
								     --maskpath hdfs:///preprocess/train/masks \
								     --outputpath hdfs:///train/train_df.parquet

PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs/bin/python spark-submit --master spark://node013.ib.cluster:7077 \
                                                                     --executor-memory 8G \
                                                                     --num-executors 16 \
                                                                     --total-executor-cores 49 \
                                                                     preprocess.py \
                                                                     --featurepath hdfs:///preprocess/validation/images \
                                                                     --maskpath hdfs:///preprocess/validation/masks \
                                                                     --outputpath hdfs:///validation/validation_df.parquet
