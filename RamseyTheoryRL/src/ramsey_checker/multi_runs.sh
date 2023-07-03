#!/bin/bash

# Python script you want to run
script="test.py"

for ((i=1; i<=10; i++))
do
   echo "Running the script for the $i time"
   python $script
done
