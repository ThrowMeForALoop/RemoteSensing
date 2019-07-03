#!/bin/bash
PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs/bin/python spark-submit --master spark://node013.ib.cluster:7077 \
 								     --executor-memory 8G \
								     --num-executors 16 \
								     --total-executor-cores 49 \
								     train_horovod.py 
#PYSPARK_PYTHON=/var/scratch/tnguyenh/anaconda3/envs/rs/bin/python spark-submit --master spark://node013.ib.cluster:7077 \
#                                                                     --executor-memory 8G \
#                                                                     --total-executor-cores 105 \
#                                                                     train_horovod.py

