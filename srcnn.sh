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

tStart=$(date +%s)
./SRCNN $1
tEnd=$(date +%s)

echo -e "  \033[0;36;1m[Info]\033[0m The program take \033[0;31;1m$((tEnd-tStart))\033[0m seconds ..."

if [ $? -eq 0 ];
then
	echo -e "  \033[0;32;1m[Running]\033[0m Complete -- Success!"
else
	echo -e "  \033[0;31;1m[Error]\033[0m Running Failed!"
	exit 6
fi

exit 0
