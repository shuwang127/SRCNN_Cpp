#!/bin/bash

# Program:
#	Compile the OpenCV project
# Arguments:
#   $1:The source file name
#	$2:The image file name
# History:
#	WangShu  Mon 14 Sep, 2015

if [ -f $1.cpp ];
then
	g++ `pkg-config --cflags opencv` -c $1.cpp
else
	echo -e "  \033[0;31;1m[Error]\033[0m $1.cpp does not exited!"
	exit 1
fi

if [ $? -eq 0 ];
then
	if [ -f $1.o ];
	then
		echo -e "  \033[0;32;1m[Compile]\033[0m Complete -- $1.o has created!"
		#g++ `pkg-config --libs opencv` -o $1 $1.o
	else
		echo -e "  \033[0;31;1m[Error]\033[0m $1.o has not created!"
		exit 2
	fi
else
	echo -e "  \033[0;31;1m[Error]\033[0m Compile Failed!"
	exit 3
fi

g++ `pkg-config --libs opencv` -o $1 $1.o

if [ $? -eq 0 ];
then
	if [ -f $1 ];
	then
		echo -e "  \033[0;32;1m[Linking]\033[0m Complete -- $1 has created!"
		#./$1
	else
		echo -e "  \033[0;31;1m[Error]\033[0m $1 has not created!"
		exit 4
	fi
else
	echo -e "  \033[0;31;1m[Error]\033[0m Linking Failed!"
	exit 5
fi

echo -e "  \033[0;36;1m[Running]\033[0m $1 is running ..."
tStart=$(date +%s)
./$1  $2
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
