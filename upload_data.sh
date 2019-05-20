#!/bin/bash
hdfs dfs -mkdir /input
hdfs dfs -mkdir /input/features
hdfs dfs -mkdir /input/masks
hdfs dfs -put /var/scratch/tnguyenh/datasets/inria_aerial/AerialImageDataset/train/images/* /input/features
hdfs dfs -put /var/scratch/tnguyenh/datasets/inria_aerial/AerialImageDataset/train/gt/* /input/masks
