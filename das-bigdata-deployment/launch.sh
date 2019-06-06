#!/bin/bash
module load python/2.7.13
module load prun

RESERVATION_ID=$(python deployer preserve create-reservation -q -t 24:00:00 8)
python deployer preserve wait-for-reservation -t 600 $RESERVATION_ID
python deployer deploy --preserve-id $RESERVATION_ID -s env/das5-hadoop.settings hadoop 2.6.0 
python deployer deploy --preserve-id $RESERVATION_ID -s env/das5-spark.settings spark 2.4.0
