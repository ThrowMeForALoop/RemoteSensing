#!/bin/bash
PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs/bin/python spark-submit --master yarn \
 								     --driver-memory 8g \
 								     --executor-memory 10g \
								     --num-executors 4 \
								     --executor-cores 8 \
								     images_to_parquet.py \
								     --inputpath hdfs:///input \
								     --outputpath hdfs:///output
