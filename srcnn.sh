#!/bin/bash

# Program:
#	Compile the OpenCV project
# Arguments:
#   $1:The image file name
# History:
#	WangShu  Mon 14 Sep, 2015

cd Result/

if [ -f Origin.bmp ];
then
	rm *.bmp
fi

cd ../
./opencv.sh SRCNN $1
