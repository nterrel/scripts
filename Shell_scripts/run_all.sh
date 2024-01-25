#!/bin/bash

testname=$(echo $1 | tr '.py' '\n' | head -n 1)
logfile=$testname"_0.log"
rm $logfile

for filename in non-periodic/* periodic/*
do
    echo $testname " test on " $filename
    echo $filename >> $logfile
    python $1 $filename >> $logfile 2>&1
done
