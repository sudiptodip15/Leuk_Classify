#!/bin/bash

python Label_DataOut.txt

python GenData.py

python Classify.py

python kMeans.py
