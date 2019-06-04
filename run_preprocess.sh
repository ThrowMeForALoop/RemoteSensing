#!/bin/bash
PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs_gpu/bin/python spark-submit --master yarn \
 								     --driver-memory 8g \
 								     --executor-memory 10g \
								     --num-executors 4 \
								     --executor-cores 8 \
								     preprocess.py \
								     --featurepath hdfs:///input/features \
								     --maskpath hdfs:///input/masks \
								     --outputpath hdfs:///output
