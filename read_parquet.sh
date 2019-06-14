#!/bin/bash
PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs/bin/python spark-submit --master spark://node013.ib.cluster:7077 \
 								     --executor-memory 8G \
								     --num-executors 16 \
								     --total-executor-cores 49 \
								     read_parquet.py \

