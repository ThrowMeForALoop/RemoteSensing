#!/bin/bash
hdfs dfs -mkdir -p /preprocess/train/images
hdfs dfs -mkdir -p /preprocess/train/masks
hdfs dfs -mkdir -p /preprocess/validation/images
hdfs dfs -mkdir -p /preprocess/validation/masks

hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/train_data_120/images/* /preprocess/train/images
hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/train_data_120/masks/* /preprocess/train/masks
hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/validation_data_120/images/* /preprocess/validation/images
hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/validation_data_120/masks/* /preprocess/validation/masks

