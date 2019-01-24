#!/bin/bash

date  > start.txt

spark-submit BottleNeckFeatureExtraction_Server.py

date  > end.txt
