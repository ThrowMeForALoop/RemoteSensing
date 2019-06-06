#!/bin/bash
hdfs dfs -mkdir -p /preprocess/train/images
hdfs dfs -mkdir -p /preprocess/train/masks
hdfs dfs -mkdir -p /preprocess/validation/images
hdfs dfs -mkdir -p /preprocess/validation/masks

hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/train_data_10000/images/* /preprocess/train/images
hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/train_data_10000/masks/* /preprocess/train/masks
hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/validation_data_2000/images/* /preprocess/validation/images
hdfs dfs -put /var/scratch/tnguyenh/src/RemoteSensing/validation_data_2000/masks/* /preprocess/validation/masks

